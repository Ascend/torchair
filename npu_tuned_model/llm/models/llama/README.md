# Llama2&Llama3

本模块主要是llama2/llama3模型在npu上的适配迁移点介绍，其中llama2使用transformers==4.31.0版本，llama3使用transformers==4.40.0版本。

# 性能优化

## 固定kv cache大小

**优化原因**：transformers的llama源码中对于kv cache的处理是作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种更新方式存在多次申请内存及拷贝的性能损失。

**优化方式**：根据句子最大长度申请好一块固定大小的kv cache tensor，然后通过[scatter_update_](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000156.html)算子对指定位置上的kv cache进行更新

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

**FlashAttention优化方式**:替换LlamaAttention中的Q/K/V和attention_mask相关的两个矩阵乘，分别替换为[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)和[torch_npu.npu_incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)

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

**cos/sin优化方式**：cos/sin原脚本中在decodelayer中每层都会计算，存在耗时浪费。提到上层，只需计算一次。旋转位置编码替换为torch_npu.npu_apply_rotary_pos_emb融合算子

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
sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
```

## Add+RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能

**优化方式**：替换LlamaRMSNorm的forward函数，使用融合算子[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_001031.html)和torch_npu.npu_add_rms_norm

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

## 全量优化计算量

优化原因：根据网络的计算逻辑，全量计算完logits后只取seq的最新位置的数据，所以在全量的最后线性层计算可以只对最新的seq位置做计算，降低计算量

优化方式：

```python
# LlamaForCausalLM forward函数计算logits的逻辑
# 修改前
if self.pretraining_tp > 1:
    lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
    logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
    logits = torch.cat(logits, dim=-1)
else:
    logits = self.lm_head(hidden_states)
# 修改后
if self.pretraining_tp > 1:
    lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
    logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
    logits = torch.cat(logits, dim=-1)
else:
    # 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，l
    # 所以在全量的最后线性层计算可以只对最新的seq位置做计算，降低计算量
    bs, seq, hidden = hidden_states.size()
    if seq > 1:
        gather_index = torch.ones(bs, dtype=torch.int64, device=hidden_states.device) * (seq - 1)
        gather_index = gather_index.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 1, hidden)
        hidden_states = torch.gather(hidden_states, 1, gather_index)
    logits = self.lm_head(hidden_states)

# torch_npu.npu_prompt_flash_attention入参next_token改成0
```

## qkv融合

优化原因：将三个矩阵乘替换为一个矩阵乘，最大化使用npu的计算能力提升性能

优化方式：

```python
# LlamaAttention __init__函数新增修改
self.qkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim + 2 * self.num_key_value_heads * 		                      self.head_dim, bias=False)

# LlamaAttention forward函数
# 修改前
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)
# 修改后，qkv融合后参与矩阵乘计算，然后将计算结果进行拆分
qkv_states = self.qkv(hidden_states)
query_states, key_states, value_states = qkv_states.split(
    [self.q_hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=2)
```

**此优化项还需要对权重进行融合，否则新创建的qkv layer没有初始权重值**

```python
# 权重融合改动
def merge_qkv_weight(self, tp_size=1):
  if self.model is None:
      raise ValueError("Model is None, please check")

  def _to_parameter(data):
      return nn.Parameter(data, requires_grad=False)

  qw_size = self.model.layers[0].self_attn.q_proj.weight.shape  # [out_channel, in_channel]
  kw_size = self.model.layers[0].self_attn.k_proj.weight.shape
  vw_size = self.model.layers[0].self_attn.v_proj.weight.shape

  q_sliced_size = qw_size[0] // tp_size
  k_sliced_size = kw_size[0] // tp_size
  v_sliced_size = vw_size[0] // tp_size
  print(f"sliced out channel size, q:{q_sliced_size}, k:{k_sliced_size}, v:{v_sliced_size}")

  for i in range(len(self.model.layers)):
      qw = self.model.layers[i].self_attn.q_proj.weight
      kw = self.model.layers[i].self_attn.k_proj.weight
      vw = self.model.layers[i].self_attn.v_proj.weight

      weight_list = []
      for j in range(tp_size):
          sliced_qw = qw[j * q_sliced_size: (j + 1) * q_sliced_size, :]
          sliced_kw = kw[j * k_sliced_size: (j + 1) * k_sliced_size, :]
          sliced_vw = vw[j * v_sliced_size: (j + 1) * v_sliced_size, :]
          weight_list.append(_to_parameter(torch.cat([sliced_qw, sliced_kw, sliced_vw], axis=0)))

      if len(weight_list) == 1:
          self.model.layers[i].self_attn.qkv.weight = weight_list[0]
      else:
          self.model.layers[i].self_attn.qkv.weight = _to_parameter(torch.cat(weight_list, axis=0))
```

## 全量替换mc2融合算子

优化原因：原生LinearAllreduce中matmul和allreduce是串行的，性能较慢。替换mc2融合算子[torch_npu.npu_mm_all_reduce_base](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000448.html)后能够使matmul和allreduce之间产生流水，提高性能。

优化方式：使用**models/common/mc2_adapter.py**自定义的**LinearAllreduce**替换原生的**deepspeed.module_inject.layers.LinearAllreduce**

```python
import torch
from torch import nn
from deepspeed import comm as dist
import torch_npu

class LinearAllreduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group
        if self.mp_group is not None:
            rank = torch.distributed.get_rank(self.mp_group)
            global_rank = torch.distributed.get_global_rank(self.mp_group, rank)
            self.hcomm_info = self.mp_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)

    def forward(self, input):
        bs, seq, hidden_size = input.shape
        if seq == 1:
            output = torch.matmul(input, self.weight.transpose(-1, -2))
            if self.mp_group is not None:
                dist.all_reduce(output, group=self.mp_group)
            if self.bias is not None:
                output += self.bias
        else:
            output = torch_npu.npu_mm_all_reduce_base(input, self.weight.transpose(-1, -2).contiguous(),
                                                      self.hcomm_info)

        return output
```

# 性能数据

## llama2-70B

执行llama2-70b，加载模型时的tensor类型float16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size：**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">优化项</th>
    <th class="tg-0pky" colspan="2">单算子模式</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">原始脚本</td>
    <td class="tg-0pky">1110ms</td>
    <td class="tg-0pky">211ms</td>
    <td class="tg-0pky">1110ms</td>
    <td class="tg-0pky">175ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">固定kv cache</td>
    <td class="tg-0pky">1110ms</td>
    <td class="tg-0pky">187ms</td>
    <td class="tg-0pky">930ms</td>
    <td class="tg-0pky">83ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">替换FlashAttention&amp;&amp;cos/sin优化</td>
    <td class="tg-0pky">984ms</td>
    <td class="tg-0pky">144ms</td>
    <td class="tg-0pky">789ms</td>
    <td class="tg-0pky">44ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">680ms</td>
    <td class="tg-0pky">120ms</td>
    <td class="tg-0pky">660ms</td>
    <td class="tg-0pky">39ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">664ms</td>
    <td class="tg-0pky">120ms</td>
    <td class="tg-0pky">633ms</td>
    <td class="tg-0pky">39ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">qkv融合(单算子精度异常)</td>
    <td class="tg-0pky">664ms</td>
    <td class="tg-0pky">120ms</td>
    <td class="tg-0pky">629ms</td>
    <td class="tg-0pky">37.5ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量替换mc2融合算子</td>
    <td class="tg-0pky">615ms</td>
    <td class="tg-0pky">120ms</td>
    <td class="tg-0pky">570ms</td>
    <td class="tg-0pky">37.5ms</td>
  </tr>
</tbody>
</table>


**800I A2不同bs的图模型性能比较：**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">batch size</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">175ms</td>
    <td class="tg-0lax">35.9ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">570ms</td>
    <td class="tg-0pky">37.5ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">1112ms</td>
    <td class="tg-0lax">38.6ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">16</td>
    <td class="tg-0lax">2325ms</td>
    <td class="tg-0lax">41.2ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">32</td>
    <td class="tg-0lax">4757ms</td>
    <td class="tg-0lax">46ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">48</td>
    <td class="tg-0pky">7118ms</td>
    <td class="tg-0pky">48.6ms</td>
  </tr>
</tbody>
</table>

## lama3-70B

执行llama3-70b，加载模型时的tensor类型float16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size：**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">优化项</th>
    <th class="tg-0pky" colspan="2">单算子模式</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">原始脚本</td>
    <td class="tg-0pky">1138ms</td>
    <td class="tg-0pky">215ms</td>
    <td class="tg-0pky">1183ms</td>
    <td class="tg-0pky">182ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">固定kv cache</td>
    <td class="tg-0pky">1134ms</td>
    <td class="tg-0pky">191ms</td>
    <td class="tg-0pky">1177ms</td>
    <td class="tg-0pky">161ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">替换FlashAttention&amp;&amp;cos/sin优化</td>
    <td class="tg-0pky">991ms</td>
    <td class="tg-0pky">148ms</td>
    <td class="tg-0pky">912ms</td>
    <td class="tg-0pky">46ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">705ms</td>
    <td class="tg-0pky">122ms</td>
    <td class="tg-0pky">679ms</td>
    <td class="tg-0pky">40ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">654ms</td>
    <td class="tg-0pky">122ms</td>
    <td class="tg-0pky">631ms</td>
    <td class="tg-0pky">40ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">qkv融合(单算子精度异常)</td>
    <td class="tg-0pky">654ms</td>
    <td class="tg-0pky">124ms</td>
    <td class="tg-0pky">620ms</td>
    <td class="tg-0pky">40ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量替换mc2融合算子</td>
    <td class="tg-0pky">618ms</td>
    <td class="tg-0pky">124ms</td>
    <td class="tg-0pky">567ms</td>
    <td class="tg-0pky">39ms</td>
  </tr>
</tbody>
</table>


**800I A2不同bs的图模型性能比较：**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">batch size</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">177.5ms</td>
    <td class="tg-0lax">37.4ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">298ms</td>
    <td class="tg-0lax">37.9ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">566.7ms</td>
    <td class="tg-0pky">38.9ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">1104.5ms</td>
    <td class="tg-0lax">40.1ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">16</td>
    <td class="tg-0lax">2324.1ms</td>
    <td class="tg-0lax">43.4ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">32</td>
    <td class="tg-0lax">4863.4ms</td>
    <td class="tg-0lax">48.2ms</td>
  </tr>
</tbody>
</table>
备注：llama3 48 Batch图模式切分8卡场景:由于llama3网络 权重（19.8543G），图分配内存（8.29G）, IO内存（3.87G），超过 B4 device内存（约29G），导致48Batch因为OOM问题无法部署。
