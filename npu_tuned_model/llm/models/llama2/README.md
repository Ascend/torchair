# Llama2

本模块主要是llama2模型在npu上的适配迁移点介绍

# 性能优化

## 固定kv cache大小

**优化原因**：transformers的llama源码中对于kv cache的处理是作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种更新方式存在多次申请内存及拷贝的性能损失。

**优化方式**：根据句子最大长度申请号一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv cache进行更新

```python
# transformers/models/llama/modeling_llama.py
# LlamaForCausalLM的prepare_inputs_for_generation函数新增逻辑
# 固定kv cache的大小，用作全量图和增量图的kv cache更新
batch_size, seq_length = input_ids.shape
use_dtype = self.model.torch_dtype
if past_key_values is None:
    kv_shape = (
        batch_size, self.model.max_position_embeddings, self.model.num_key_value_heads // self.world_size,
        self.model.hidden_size // self.model.num_attention_heads)
    past_key_values = ()
    for i in range(self.model.num_hidden_layers):
        k_cache = torch.zeros(kv_shape, dtype=use_dtype, device=input_ids.device)
        v_cache = torch.zeros(kv_shape, dtype=use_dtype, device=input_ids.device)
        past_key_values += ((k_cache, v_cache),)
```

更新kv的改动

```python
# 更新指定位置上的kv cache，position_ids在全量图执行时从seq_len 0的位置更新，在增量图执行时从seq_len位置更新
tmp_ids = updated_kv_positions.reshape(-1)
# format BSND, 1 means seq_len dim index
torch_npu.scatter_update_(past_key_value[0], tmp_ids, key_states, 1)
torch_npu.scatter_update_(past_key_value[1], tmp_ids, value_states, 1)

key_states1 = past_key_value[0] if q_len == 1 else key_states
value_states1 = past_key_value[1] if q_len == 1 else value_states

past_key_value = past_key_value if use_cache else None
```

固定kv后，由于shape变化带来的其他tensor的改动

```python
# prepare_inputs_for_generation函数中新增创建attention_mask以及更新kv位置tensor
# 主要原因是全量和增量流程对于attention_mask的shape要求不一样，kv使用scatter_update更新需要指定更新位置
past_key_values_length = 0
if seq_length > 1:
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
    self.padding_mask = torch.zeros(batch_size, self.model.max_position_embeddings, device=input_ids.device)
    self.prompt_length = seq_length
    self.updated_kv_positions = torch.zeros(batch_size, dtype=position_ids.dtype, device=position_ids.device)
else:
    bsz, src_len = attention_mask.size()
    padding_mask = self.padding_mask
    padding_mask[:, :src_len] = attention_mask
    attention_mask = padding_mask
    past_key_values_length = self.model.max_position_embeddings
    self.prompt_length += 1
    self.updated_kv_positions = torch.ones(position_ids.shape, dtype=position_ids.dtype,
                                           device=position_ids.device) * (self.prompt_length - 1)

attention_mask = self.model._prepare_decoder_attention_mask(
    attention_mask, (batch_size, seq_length), past_key_values[0][0], past_key_values_length
)
```

## 替换FlashAttention&&cos/sin优化

**优化原因**：将小算子替换为融合大算子，提升性能

**FlashAttention优化方式**:替换LlamaAttention中的Q/K/V和attention_mask相关的两个矩阵乘

```python
''' 替换部分
# repeat k/v heads if n_kv_heads < n_heads
key_states1 = repeat_kv(key_states1, self.num_key_value_groups)
value_states1 = repeat_kv(value_states1, self.num_key_value_groups)

attn_weights = torch.matmul(query_states, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)

if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    raise ValueError(
        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        f" {attn_weights.size()}"
    )

if attention_mask is not None:
    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        )
    attn_weights = attn_weights + attention_mask

# upcast attention to fp32
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states1)

if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        f" {attn_output.size()}"
    )

attn_output = attn_output.transpose(1, 2).contiguous()
'''

# 替换后
attention_mask = attention_mask.to(torch.bool)
if q_len > 1:
    attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states1.contiguous(),
                                                       value_states1.contiguous(), num_heads=self.num_heads,
                                                       input_layout="BSND",
                                                       scale_value=self.scale_value,
                                                       pre_tokens=65535, next_tokens=65535,
                                                       atten_mask=attention_mask,
                                                       num_key_value_heads=self.num_key_value_heads)
else:
    attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states1.contiguous(),
                                                       value_states1.contiguous(), num_heads=self.num_heads,
                                                       input_layout="BSND",
                                                       scale_value=self.scale_value,
                                                       atten_mask=attention_mask,
                                                       num_key_value_heads=self.num_key_value_heads)
```

**cos/sin优化方式**：cos/sin原脚本中在decodelayer中每层都会计算，存在耗时浪费。提到上层，只需计算一次。

```python
# step1 将LlamaAttention::_init_rope函数移到LlamaModel并且到LlamaModel::__init__中调用

# step2 新增LlamaModel::_prepare_decoder_rotray_cos_sin
def _prepare_decoder_rotary_cos_sin(self, position_ids):
  cos, sin = self.rotary_emb()
  cos = cos.squeeze(1).squeeze(0)
  sin = sin.squeeze(1).squeeze(0)
  f_position_ids = position_ids.flatten()
  cos = torch.index_select(cos, 0, f_position_ids)
  sin = torch.index_select(sin, 0, f_position_ids)
  cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
  sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
  return cos, sin
# step 3 在LlamaModel的forward函数中计算cos，sin并传递到LlamaDecoderLayer::forward，LlamaAttention::forward
rotary_emb_cos, rotary_emb_sin = self._prepare_decoder_rotary_cos_sin(position_ids)

# step 4 LlamaAttention::forward函数中使用计算好的cos,sin
def apply_rotary_pos_emb(q, k, cos, sin):
    return torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin)

query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                rotary_emb_cos.to(value_states.dtype),
                                                rotary_emb_sin.to(value_states.dtype))

# step 5 删除LlamaAttention中_init_rope()和计算cos, sin相关操作

# cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```

带来的其他改动点：

- 创建attention_mask时两处torch.full填充的值由min改成1

```python
# _make_causal_mask函数
mask = torch.full((tgt_len, tgt_len), 1, device=device)
# _expand_mask函数
inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)
```

- 由于FlashAttention算子layout为BSND，消除transpose

```python
# step1 Q,K,V消除transpose
query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

# step2 kv_cache shape调整shape， scatter_update的dim
# prepare_inputs_for_generation函数
kv_shape = (batch_size, self.model.max_position_embeddings, self.model.num_key_value_heads // self.world_size, self.model.hidden_size // self.model.num_attention_heads)
# LlamaAttention::forward函数
torch_npu.scatter_update_(past_key_value[0], tmp_ids, key_states, 1)
torch_npu.scatter_update_(past_key_value[1], tmp_ids, value_states, 1)

# step3 cos，sin调整unsqueeze的dim
# _prepare_decoder_rotary_cos_sin函数
cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)Add_RMSNorm融合
```

## Add+RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能

**优化方式**：替换LlamaRMSNorm的forward函数

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self,
                hidden_states,
                residual: Optional[torch.Tensor] = None):
        if residual is None:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states
        else:
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
        return y, x
```

带来的其他地方的改动：在LlamaDecoderLayer和LlamaModel中都使用到了LlamaRMSNorm

```python
# LlamaDecoderLayer forward函数变动，先注释掉原先residual相关的代码
hidden_states, residual = self.input_layernorm(hidden_states， past_residual)
hidden_states， residual = self.post_attention_layernorm(hidden_states, residual)
outputs = (residual, hidden_states,)

# LlamaModel forward函数变动
residual = None
layer_outputs = decoder_layer(
                    hidden_states,
                    past_residual=residual,
                    attention_mask=attention_mask,
                    updated_kv_positions=updated_kv_positions,
                    rotary_emb_cos=rotary_emb_cos,
                    rotary_emb_sin=rotary_emb_sin,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    )
residual = layer_outputs[0]
hidden_states = layer_outputs[1]
if use_cache:
    next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

    if output_attentions:
        all_self_attns += (layer_outputs[2],)

hidden_states = self.norm(hidden_states, residual)
```