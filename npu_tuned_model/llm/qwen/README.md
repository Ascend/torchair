# Qwen2

本模块主要是Qwen2模型在npu上的适配迁移点介绍，本次适配基于transformers==4.37.0，模型文件在transformers目录下的models/qwen2/modeling_qwen2.py。
# 下载模型权重

Qwen2目前支持5个尺寸的预训练和指令微调模型，包括Qwen2-0.5B、Qwen2-1.5B、Qwen2-7B、Qwen2-57B-A14以及Qwen2-72B。

模型介绍：https://qwenlm.github.io/zh/blog/qwen2

权重下载链接：https://huggingface.co/Qwen

# 图模式适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：需先保证模型在npu上的eager模式功能正常和精度正确，然后再进行以下图模式的迁移和适配。

此适配点主要是新增pytorch图模式分支。

```python
# transformers/generation/utils.py中greedy_search函数while True前添加：
import os
import time
import logging

exe_mode = os.getenv("EXE_MODE", "dynamo")
dynamic_compile = eval(os.getenv("DYNAMIC_COMPILE", "False"))

if exe_mode == "dynamo":
    logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic_compile,
                                                                                               True))
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    self = torch.compile(self, dynamic=dynamic_compile, fullgraph=True, backend=npu_backend)
else:
    logging.info("Start to run model in eager(HOST API) mode")
  
while True:
    if synced_gpus:
        ...
  
# 在while true函数中模型执行前后添加torch.npu.synchronize()，主要是做性能统计，打点位置参考如下：
torch.npu.synchronize()
start = time.time()
outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
torch.npu.synchronize()
end = time.time()
cost = end -start
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能；同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力，根据dynamic参数决定是否走动态图。

**注**：若使用deepspeed框架，需要额外导入hccl算子入图

```python
import torch_npu
import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
```

```python
# 修改Qwen2Model中attention_mask的入参sliding_window。sliding_window会使attention_mask的shape不固定且无法泛化，导致增量图重复编译。
sliding_window = self.config.sliding_window if self.config.use_sliding_window else None
attention_mask = _prepare_4d_causal_attention_mask(
    attention_mask,
    (batch_size, seq_length),
    inputs_embeds,
    past_key_values_length,
    sliding_window=sliding_window,
)
  
""" 替换部分
attention_mask = _prepare_4d_causal_attention_mask(
    attention_mask,
    (batch_size, seq_length),
    inputs_embeds,
    past_key_values_length,
    sliding_window=self.config.sliding_window,
)
"""
```

```python
# 修改Qwen2Attention中forward逻辑，避免oom问题
if q_len > 1:
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
else:
    attn_output = attn_output.reshape(bsz, self.hidden_size)

attn_output = self.o_proj(attn_output)

if q_len == 1:
    attn_output = attn_output.unsqueeze(1)

""" 替换部分
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

attn_output = self.o_proj(attn_output)
"""
```

# 性能优化

**注：** 在modeling_qwen2.py文件中，被修改的原函数都加了'_'前缀，可用于对比修改后的函数变化。下面列出了优化项主要的改动，不同的大模型适配完主要改动后，再根据模型实际代码逻辑进行调试。
如果期望使用的模型是qwen1.5和qwen2，可以直接使用改造后的modeling_qwen2.py。

## 固定kv cache大小

**优化原因** transformers的qwen2源码中kv cache作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种
更新方式存在多次申请内存及拷贝的性能损失。

优化方式：根据句子最大长度申请好一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv_cache进行更新。

```python
# modeling_qwen2.py中Qwen2Model创建past_kv_values的逻辑替换如下：
# 固定kv cache的大小，用作全量图和增量图的kv cache更新
batch_size, seq_length = input_ids.shape
if use_cache:
    use_legacy_cache = not isinstance(past_key_values, Cache)
    if use_legacy_cache:
        kv_shape = (
            batch_size, self.config.num_key_value_heads // self.config.world_size,
            self.config.max_position_embeddings,
            self.config.hidden_size // self.config.num_attention_heads)
        past_key_values = ()
        for layer_idx in range(self.config.num_hidden_layers):
            k_cache = torch.zeros(kv_shape, dtype=self.config.torch_dtype, device=input_ids.device)
            v_cache = torch.zeros(kv_shape, dtype=self.config.torch_dtype, device=input_ids.device)
            past_key_values += ((k_cache, v_cache),)
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

past_key_values_length = self.max_position_embeddings if seq_length == 1 else 0
```

更换kv cache的改动：

```python
# 更新指定位置上的kv cache，position_ids在全量图执行时从起始位置更新，在增量图执行时从seq_len位置更新
if use_cache and past_key_value is not None:
    tmp_ids = updated_kv_positions.reshape(-1)
    torch_npu.scatter_update_(past_key_value.key_cache[self.layer_idx], tmp_ids, key_states, 2)
    torch_npu.scatter_update_(past_key_value.value_cache[self.layer_idx], tmp_ids, value_states, 2)
    kv_states = past_key_value[self.layer_idx] if q_len == 1 else (key_states, value_states)
    key_states = kv_states[0]
    value_states = kv_states[1]
```

固定kv cache大小后，由于shape变化带来的其他tensor的改动点：

```python
# prepare_inputs_for_generation函数改动点如下：
# 增量图输入长度固定为1；将past_key_values从Tuple[Tuple[torch.FloatTensor]]类型转换为DynamicCache类型
prompt_length = input_ids.shape[1]
# only last token for input_ids if past is not None
if past_key_values:
    input_ids = input_ids[:, -1:]
    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
input_ids = input_ids.clone()


# 将创建attention_mask的逻辑从Qwen2Model中移到prepare_inputs_for_generation中
# 根据全量图和增量图对attention_mask的shape要求不同以及kv cache更新的位置不同，创建不同的attention_mask和updated_kv_positions
batch_size, seq_length = input_ids.shape
past_key_values_length = self.config.max_position_embeddings if seq_length == 1 else 0
if seq_length > 1:
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
    updated_kv_positions = torch.zeros(batch_size, dtype=position_ids.dtype, device=position_ids.device)
else:
    bsz, src_len = attention_mask.size()
    padding_mask = torch.zeros(batch_size, self.config.max_position_embeddings, device=input_ids.device)
    padding_mask[:, :src_len] = attention_mask
    attention_mask = padding_mask
    updated_kv_positions = torch.ones(position_ids.shape, dtype=position_ids.dtype,
                                      device=position_ids.device) * (prompt_length - 1)

if inputs_embeds is None:
    inputs_embeds = self.model.embed_tokens(input_ids)
attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds,
                                                   past_key_values_length)
```

## 替换FlashAttention && cos/sin计算优化

**优化原因:** 将小算子替换为融合大算子，提升计算性能。

**FlashAttention优化方式：** 替换Qwen2Attention中q/k/v和attention_mask相关的两个矩阵乘，分别替换为
[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)和[torch_npu.npu_incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)。

```python
""" 替换部分
# repeat k/v heads if n_kv_heads < n_heads
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)

attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

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
attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
attn_output = torch.matmul(attn_weights, value_states)

if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    raise ValueError(
        f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        f" {attn_output.size()}"
    )

attn_output = attn_output.transpose(1, 2).contiguous()
"""

# 替换后
attention_mask = attention_mask.to(torch.bool)
if q_len > 1:
    attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states.contiguous(),
                                                       value_states.contiguous(), num_heads=self.num_heads,
                                                       input_layout="BSND",
                                                       scale_value=self.scale_value,
                                                       pre_tokens=65535, next_tokens=65535,
                                                       atten_mask=attention_mask,
                                                       num_key_value_heads=self.num_key_value_heads)
else:
    attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states.contiguous(),
                                                      value_states.contiguous(), num_heads=self.num_heads,
                                                      input_layout="BSND",
                                                      scale_value=self.scale_value,
                                                      atten_mask=attention_mask,
                                                      num_key_value_heads=self.num_key_value_heads)
```

**cos/sin计算优化方式：** cos/sin原脚本中在decodelayer中每层都会计算，存在耗时浪费。提到上层，只需计算一次。
旋转位置编码替换为torch_npu.npu_apply_rotary_pos_emb融合算子。

```python
# step1：将Qwen2Attention中的Qwen2RotaryEmbedding函数移到Qwen2Model并且到Qwen2Model::__init__中调用

# step2：在Qwen2Model中新增函数_prepare_decoder_rotary_cos_sin
def _prepare_decoder_rotary_cos_sin(self, position_ids):
    cos, sin = self.rotary_emb()
    f_position_ids = position_ids.flatten()
    cos = torch.index_select(cos, 0, f_position_ids)
    sin = torch.index_select(sin, 0, f_position_ids)
    cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
    sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
    return cos, sin

# step3：在Qwen2Model的forward函数中计算cos、sin并传递到Qwen2DecoderLayer::forward、Qwen2Attention::forward
rotary_emb_cos, rotary_emb_sin = self._prepare_decoder_rotary_cos_sin(position_ids)

# step4：替换apply_rotary_pos_emb函数
def apply_rotary_pos_emb(q, k, cos, sin):
    return torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin)

# step5：Qwen2Attention::forward函数中使用计算好的cos、sin
query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                rotary_emb_cos.to(value_states.dtype),
                                                rotary_emb_sin.to(value_states.dtype))
```

带来的其他改动点：

- 创建attention_mask时两处torch_full填充的值由min改为1

```python
# transformers/modeling_attn_mask_utils.py文件
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
# Qwen2Attention::forward函数
torch_npu.scatter_update_(past_key_value.key_cache[self.layer_idx], tmp_ids, key_states, 1)
torch_npu.scatter_update_(past_key_value.value_cache[self.layer_idx], tmp_ids, value_states, 1)

# step3 cos，sin调整unsqueeze的dim
# _prepare_decoder_rotary_cos_sin函数
cos = cos.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
sin = sin.reshape(position_ids.size(0), position_ids.size(1), -1).unsqueeze(2)
```

## Add+RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能。

**优化方式**：替换Qwen2RMSNorm的forward函数，使用融合算子[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_001031.html)和torch_npu.npu_add_rms_norm。

```python
def forward(self,
            hidden_states,
            residual: Optional[torch.Tensor] = None):
    if residual is None:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states
    else:
        y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
    return y, x
```

带来的其他地方的改动：在Qwen2DecoderLayer和Qwen2Model中都使用到了Qwen2RMSNorm

```python
# Qwen2DecoderLayer forward函数变动，先注释掉原先residual相关的代码
hidden_states, residual = self.input_layernorm(hidden_states， past_residual)
hidden_states， residual = self.post_attention_layernorm(hidden_states, residual)
outputs = (residual, hidden_states,)

# Qwen2Model forward函数变动
residual = None
for decoder_layer in self.layers:
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if self.gradient_checkpointing and self.training:
        layer_outputs = self._gradient_checkpointing_func(
            decoder_layer.__call__,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
        )
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            past_residual=residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            updated_kv_positions=updated_kv_positions,
            rotary_emb_cos=rotary_emb_cos,
            rotary_emb_sin=rotary_emb_sin,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

    residual = layer_outputs[0]
    hidden_states = layer_outputs[1]

    if use_cache:
        next_decoder_cache = layer_outputs[3 if output_attentions else 2]

    if output_attentions:
        all_self_attns += (layer_outputs[2],)

hidden_states, _ = self.norm(hidden_states, residual)

# add hidden states from the last decoder layer
if output_hidden_states:
    all_hidden_states += (hidden_states,)

next_cache = None
if use_cache:
    next_cache = next_decoder_cache.to_legacy_cache()
```

## 全量优化计算量

优化原因：根据网络的计算逻辑，全量计算完logits后只取seq的最新位置的数据，因此在全量的最后线性层只需对最新的seq位置做计算，降低计算量

优化方式：

```python
# Qwen2ForCausalLM forward函数计算logits的逻辑
# 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，所以在全量的最后线性层只需对最新的seq位置做计算
bs, seq, hidden = hidden_states.size()
if seq > 1:
    gather_index = torch.ones(bs, dtype=torch.int64, device=hidden_states.device) * (seq - 1)
    gather_index = gather_index.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, 1, hidden)
    hidden_states = torch.gather(hidden_states, 1, gather_index)
logits = self.lm_head(hidden_states)
logits = logits.float()

# torch_npu.npu_prompt_flash_attention入参next_token改成0
```

## qkv融合

优化原因：将三个矩阵乘替换为一个矩阵乘，最大化使用npu的计算能力提升性能

优化方式：

```python
# Qwen2Attention __init__函数新增修改
self.qkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim, bias=True)

# Qwen2Attention forward函数
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

    qw_sliced_size = qw_size[0] // tp_size
    kw_sliced_size = kw_size[0] // tp_size
    vw_sliced_size = vw_size[0] // tp_size
    print(
        f"sliced out channel size, q_weight:{qw_sliced_size}, k_weight:{kw_sliced_size}, v_weight:{vw_sliced_size}")

    qbias_size = self.model.layers[0].self_attn.q_proj.bias.shape  # [out_channel]
    kbias_size = self.model.layers[0].self_attn.k_proj.bias.shape
    vbias_size = self.model.layers[0].self_attn.v_proj.bias.shape

    qbias_sliced_size = qbias_size[0] // tp_size
    kbias_sliced_size = kbias_size[0] // tp_size
    vbias_sliced_size = vbias_size[0] // tp_size
    print(
        f"sliced out channel size, q_bias:{qbias_sliced_size}, k_bias:{kbias_sliced_size}, v_bias:{vbias_sliced_size}")

    for i in range(len(self.model.layers)):
        qw = self.model.layers[i].self_attn.q_proj.weight
        kw = self.model.layers[i].self_attn.k_proj.weight
        vw = self.model.layers[i].self_attn.v_proj.weight

        qbias = self.model.layers[i].self_attn.q_proj.bias
        kbias = self.model.layers[i].self_attn.k_proj.bias
        vbias = self.model.layers[i].self_attn.v_proj.bias

        weight_list = []
        bias_list = []
        for j in range(tp_size):
            sliced_qw = qw[j * qw_sliced_size: (j + 1) * qw_sliced_size, :]
            sliced_kw = kw[j * kw_sliced_size: (j + 1) * kw_sliced_size, :]
            sliced_vw = vw[j * vw_sliced_size: (j + 1) * vw_sliced_size, :]
            weight_list.append(_to_parameter(torch.cat([sliced_qw, sliced_kw, sliced_vw], axis=0)))

            sliced_qbias = qbias[j * qbias_sliced_size: (j + 1) * qbias_sliced_size]
            sliced_kbias = kbias[j * kbias_sliced_size: (j + 1) * kbias_sliced_size]
            sliced_vbias = vbias[j * vbias_sliced_size: (j + 1) * vbias_sliced_size]
            bias_list.append(_to_parameter(torch.cat([sliced_qbias, sliced_kbias, sliced_vbias], axis=0)))

        if len(weight_list) == 1:
            self.model.layers[i].self_attn.qkv.weight = weight_list[0]
            self.model.layers[i].self_attn.qkv.bias = bias_list[0]
        else:
            self.model.layers[i].self_attn.qkv.weight = _to_parameter(torch.cat(weight_list, axis=0))
            self.model.layers[i].self_attn.qkv.bias = _to_parameter(torch.cat(bias_list, axis=0))
```

## IFA算子计算优化+tiling全下沉

优化原因：IFA算子在进行增量计算时，由于输出做了padding，导致存在冗余的计算，可以通过传入actual_seq_lengths和kv_padding_size参数
来消除冗余的计算。同时使能tiling全下沉配置项，将IFA算子的tiling计算转移到AICPU侧异步执行，减少host-device间的同步和调度时延。

优化方式：
```python
# Step1：prepare_inputs_for_generation新增模型入参actual_seq_len和kv_padding_size
kv_padding_size = torch.tensor(self.config.max_position_embeddings - prompt_length, device=position_ids.device)
actual_seq_len = (position_ids[:, -1] + 1).cpu().tolist()


# Step2：使能IFA算子的actual_seq_lengths和kv_padding_size后，模型需要走动态模式，但模型实际输入为静态，
# 所以prepare_inputs_for_generation中需要对输入进行mark_static
self._mark_model_inputs_static(model_inputs)

def _mark_model_inputs_static(self, model_inputs):
    for key, value in model_inputs.items():
        if key == "past_key_values" and value is not None:
            for i in range(self.config.num_hidden_layers):
                torch._dynamo.mark_static(value[i][0])
                torch._dynamo.mark_static(value[i][1])
        elif isinstance(value, torch.Tensor):
            torch._dynamo.mark_static(value)

            
# Step3：Qwen2Attention中torch_npu.npu_incre_flash_attention算子新增参数actual_seq_lengths和kv_padding_size：
attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states.contiguous(),
                                                  value_states.contiguous(), num_heads=self.num_heads,
                                                  input_layout="BSND",
                                                  scale_value=self.scale_value,
                                                  atten_mask=attention_mask,
                                                  actual_seq_lengths=actual_seq_len,      # 新增
                                                  kv_padding_size=kv_padding_size,        # 新增
                                                  num_key_value_heads=self.num_key_value_heads)
```

**开启tiling全下沉配置，并将torch.compile的dynamic参数配置为True**
```python
# 将transformers/generation/utils.py中greedy_search函数中的图模式代码改为：
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
config = CompilerConfig()
config.experimental_config.frozen_parameter = True
config.experimental_config.tiling_schedule_optimize = True    # 使能tiling全下沉配置
npu_backend = tng.get_npu_backend(compiler_config=config)
self = torch.compile(self, dynamic=True, fullgraph=True, backend=npu_backend)    # 将dynamic参数配置为True
```


# 性能数据

## Qwen2-7B

执行Qwen2-7B，加载模型时的tensor类型bfloat16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size，单卡性能数据：**

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
    <td class="tg-0pky">575ms</td>
    <td class="tg-0pky">52ms</td>
    <td class="tg-0pky">585ms</td>
    <td class="tg-0pky">44ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">固定kv cache</td>
    <td class="tg-0pky">577ms</td>
    <td class="tg-0pky">53ms</td>
    <td class="tg-0pky">545ms</td>
    <td class="tg-0pky">41ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">替换FlashAttention&amp;&amp;cos/sin优化</td>
    <td class="tg-0pky">403ms</td>
    <td class="tg-0pky">33ms</td>
    <td class="tg-0pky">389ms</td>
    <td class="tg-0pky">26ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">385ms</td>
    <td class="tg-0pky">27ms</td>
    <td class="tg-0pky">369ms</td>
    <td class="tg-0pky">24ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">361ms</td>
    <td class="tg-0pky">25ms</td>
    <td class="tg-0pky">345ms</td>
    <td class="tg-0pky">24ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">qkv融合</td>
    <td class="tg-0pky">366ms</td>
    <td class="tg-0pky">25ms</td>
    <td class="tg-0pky">345ms</td>
    <td class="tg-0pky">24ms</td>
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
    <td class="tg-0lax">94ms</td>
    <td class="tg-0lax">23ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">345ms</td>
    <td class="tg-0pky">24ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">636ms</td>
    <td class="tg-0lax">29ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">16</td>
    <td class="tg-0lax">1244ms</td>
    <td class="tg-0lax">33ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">32</td>
    <td class="tg-0lax">2496ms</td>
    <td class="tg-0lax">36ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">48</td>
    <td class="tg-0pky">3692ms</td>
    <td class="tg-0pky">45ms</td>
  </tr>
</tbody>
</table>

## Qwen2-72B

执行Qwen2-72B，加载模型时的tensor类型bfloat16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size, 8卡性能数据：**

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
    <td class="tg-0pky">1179ms</td>
    <td class="tg-0pky">207ms</td>
    <td class="tg-0pky">1228ms</td>
    <td class="tg-0pky">165ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">固定kv cache</td>
    <td class="tg-0pky">1185ms</td>
    <td class="tg-0pky">206ms</td>
    <td class="tg-0pky">1092ms</td>
    <td class="tg-0pky">65ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">替换FlashAttention&amp;&amp;cos/sin优化</td>
    <td class="tg-0pky">1043ms</td>
    <td class="tg-0pky">149ms</td>
    <td class="tg-0pky">953ms</td>
    <td class="tg-0pky">47ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">760ms</td>
    <td class="tg-0pky">118ms</td>
    <td class="tg-0pky">724ms</td>
    <td class="tg-0pky">41ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">703ms</td>
    <td class="tg-0pky">125ms</td>
    <td class="tg-0pky">708ms</td>
    <td class="tg-0pky">41ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">qkv融合(单算子精度异常)</td>
    <td class="tg-0pky">698ms</td>
    <td class="tg-0pky">122ms</td>
    <td class="tg-0pky">676ms</td>
    <td class="tg-0pky">38ms</td>
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
    <td class="tg-0lax">224ms</td>
    <td class="tg-0lax">38ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">2</td>
    <td class="tg-0lax">360ms</td>
    <td class="tg-0lax">38ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">676ms</td>
    <td class="tg-0pky">38ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">1259ms</td>
    <td class="tg-0lax">39ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">16</td>
    <td class="tg-0lax">2437ms</td>
    <td class="tg-0lax">42ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">32</td>
    <td class="tg-0lax">4920ms</td>
    <td class="tg-0lax">47ms</td>
  </tr>
</tbody>
</table>

**2024年8月20号，4batch新增性能数据：**
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
    <td class="tg-0pky">IFA算子计算优化+tiling全下沉</td>
    <td class="tg-0pky">689ms</td>
    <td class="tg-0pky">122ms</td>
    <td class="tg-0pky">688ms</td>
    <td class="tg-0pky">37ms</td>
  </tr>
</tbody>
</table>

# 性能测试

## 环境准备和代码适配

在benchmark目录下的deepspeed目录提供了对接deepspeed框架多卡切分的qwen2执行样例参考。上述性能数据基于deepspeed(0.14.2)在arm host + 800I A2环境执行进行统计

**基于搭建的conda环境，安装对应的transformers版本**

```shell
# qwen2
pip3 install transformers==4.37.0
```

**设置环境变量**

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/llm/qwen
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
```

**代码适配**

```shell
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}')
# 1.根据图模式适配方式修改${transformers_path}/transformers/generation/utils.py中的函数

# 2.根据“FlashAttention && cos/sin计算优化”替换${transformers_path}/transformers/modeling_attn_mask_utils.py中的函数
# _make_causal_mask函数
mask = torch.full((tgt_len, tgt_len), 1, device=device)
# _expand_mask函数
inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)
```

## 单卡性能测试（以Qwen2-7B跑单卡为例）

**Step1：qkv权重融合**

```shell
model_path=xxx/Qwen2-7B # 下载的权重和模型信息
python3 merge_qkv_weight.py --model_path=${model_path} --output_path=xxx/Qwen2-7B-qkv
```

**Step2：执行用例脚本**

```shell
# 图模式
python -u benchmark/npu/benchmark_qwen.py --model_path=xxx/Qwen2-7B-qkv
# 单算子
python -u benchmark/npu/benchmark_qwen.py --model_path=xxx/Qwen2-7B-qkv --execute_mode=eager
```

## 多卡性能测试（以Qwen2-72B跑8卡为例）

**Step1：qkv权重融合**

```shell
model_path=xxx/Qwen2-72b # 下载的权重和模型信息
python3 merge_qkv_weight.py --model_path=${model_path} --tp_size=8 --output_path=xxx/Qwen2-72b-qkv
```

**Step2：deepspeed方式拉起8卡执行**

```shell
# 图模式
deepspeed --num_gpus=8 benchmark/deepspeed/benchmark_qwen.py --model_path=xxx/Qwen2-72b-qkv
# 单算子
deepspeed --num_gpus=8 benchmark/deepspeed/benchmark_qwen.py --model_path=xxx/Qwen2-72b-qkv --execute_mode=eager
```

**性能数据打点位置**

参考图模式适配模型执行部分修改。全量耗时统计的是模型第一个step数据，增量耗时统计后续step数据的均值。