# Qwen2-VL

本模块主要是qwen2-vl模型在npu上的适配迁移点介绍，其中transformers使用4.47.0版本，模型是在对于的transformers目录下的models/qwen2_vl/modeling_qwen2_vl.py。
# 下载模型权重

模型介绍：https://github.com/QwenLM/Qwen2-VL/tree/master

权重下载链接：https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 图模式适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：需先保证模型在npu上的eager模式功能正常和精度正确，然后再进行以下图模式的迁移和适配。

为了方便后续图模式适配以及性能优化项的改造，需要先对模型结构进行调整，并把部分不支持入图的算子挪到模型的输入准备阶段：

此适配点主要是新增pytorch图模式分支。

CompilerConfig配置参考[torchair资料](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/torchairuseguide/torchair_0021.html)

```python
# transformers/generation/utils.py的greedy_search函数while True前添加
import os
import time
import logging

exe_mode = os.getenv("EXE_MODE", "dynamo")
dynamic_compile = True # 因为当模型结构使能了actual_seq_length

if exe_mode == "dynamo":
    logging.info("Start to run model in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic_compile, True))
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True # tiling全下沉性能优化
    npu_backend = tng.get_npu_backend(compiler_config=config)
    self = torch.compile(self, dynamic=dynamic_compile, fullgraph=True, backend=npu_backend)
else:
    logging.info("Start to run model in eager(HOST API) mode")

while True:
    if synced_gpus:
        ...

# 在模型执行前后添加torch.npu.synchronize()，主要是做性能统计，打点位置参考如下
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
cost = end - start # 每个step的耗时统计
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能。同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力。根据dynamic参数决定是否走动态图。

```python
# ① 将rotary_pos_emb的计算从Qwen2VisionTransformerPretrainedModel中挪到Qwen2VLForConditionalGeneration的prepare_inputs_for_generation函数中：
# 新增函数rot_pos_emb，并在prepare_inputs_for_generation中调用：
def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
# 在self.visual中新增1个参数rotary_pos_emb：
video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw, rotary_pos_emb=rotary_pos_emb)

# ② 将VIT中cu_seqlens和attention_mask的计算从Qwen2VisionTransformerPretrainedModel和VisionAttention中挪到Qwen2VLForConditionalGeneration的forward函数中：
# cu_seqlens计算移至forward函数中：
cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    dim=0,
    # Select dtype based on the following factors:
    #  - FA2 requires that cu_seqlens_q must have dtype int32
    #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
    # See https://github.com/huggingface/transformers/pull/34852 for more information
    dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
# attention_mask计算移至forward函数中
embed_hidden_states = self.patch_embed(pixel_values) # 传入VisionAttention中的hidden_states经过第一次计算
embed_hidden_states = self.norm(embed_hidden_states) # 传入VisionAttention中的hidden_states经过第二次计算
seq_length = embed_hidden_states.shape[0]
embed_attention_mask = torch.full(
    [1, seq_length, seq_length],
    torch.finfo(embed_hidden_states.dtype).min,
    device=embed_hidden_states.device,
    dtype=embed_hidden_states.dtype
)
for i in range(1, len(cu_seqlens)):
    embed_attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
# 在self.visual中新增2个参数cu_seqlens和attention_mask：
video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw, rotary_pos_emb=rotary_pos_emb,
    cu_seqlens=cu_seqlens, attention_mask=embed_attention_mask)


# ③ 新增图模式接口，并在prepare_inputs_for_generation中调用：
def check_execute_mode(self):
    exe_mode = os.getenv("EXE_MODE", "dynamo")
    global compiled
    if exe_mode == "dynamo" and not compiled:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        config = tng.CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.experimental_config.tiling_schedule_optimize = True
        npu_backend = tng.get_npu_backend(compiler_config=config)
        self.visual = torch.compile(self.visual, dynamic=True, fullgraph=True, backend=npu_backend)
        self.model = torch.compile(self.model, dynamic=True, fullgraph=True, backend=npu_backend)
        compiled = True
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能；同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力，根据dynamic参数决定是否走动态图。
- 由于模型结构发生了变化，需要对权重文件做对应的转换，避免权重加载报错：

# 性能优化

**注：** 在vision_process.py、modeling_qwen2_vl.py和image_processing_qwen2_vl.py文件中，被修改的原函数都加了'_'前缀，可用于对比修改后的函数变化。下面列出了优化项主要的改动，不同的大模型适配完主要改动后，再根据模型实际代码逻辑进行调试。
如果期望使用的模型是Qwen2-VL，可以直接使用改造后的vision_process.py、modeling_qwen2_vl.py和image_processing_qwen2_vl.py。

## 固定kv cache大小

**优化原因** modeling_qwen2.py的源码中kv cache作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种
更新方式存在多次申请内存及拷贝的性能损失。

优化方式：根据句子最大长度申请好一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv_cache进行更新。

```python
# modeling_qwen.py中Qwen2VLModel创建past_kv_values的逻辑挪到Qwen2VLForConditionalGeneration的prepare_inputs_for_generation函数中:
# 新增函数prepare_kv_cache，固定kv cache的大小，用作全量图和增量图的kv cache更新，在prepare_inputs_for_generation函数中调用
def prepare_kv_cache(self, input_ids, past_key_values, attention_mask, position_ids, prompt_length):
    # 固定kv cache的大小，用作全量图和增量图的kv cache更新
    batch_size, seq_length = input_ids.shape
    if past_key_values is None:
        kv_shape = (
            batch_size, self.config.max_position_embeddings, self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads)
        past_key_values = ()
        for i in range(self.config.num_hidden_layers):
            k_cache = torch.zeros(kv_shape, dtype=self.config.torch_dtype, device=input_ids.device)
            v_cache = torch.zeros(kv_shape, dtype=self.config.torch_dtype, device=input_ids.device)
            past_key_values += ((k_cache, v_cache),)

    # 增量图attention_mask padding到最大长度
    # 增加updated_kv_positions给固定kv cache的tensor更新提供更新位置
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
    attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
```

更新kv cache的改动：

```python
# 更新指定位置上的kv cache，position_ids在全量图执行时从起始位置更新，在增量图执行时从seq_len位置更新
# Qwen2VLAttention::forward函数中更新kv cache的逻辑改动如下：
if use_cache and past_key_value is not None:
    tmp_ids = updated_kv_positions.reshape(-1)
    torch_npu.scatter_update_(past_key_value.key_cache[self.layer_idx], tmp_ids[-1:], key_states, 2)
    torch_npu.scatter_update_(past_key_value.value_cache[self.layer_idx], tmp_ids[-1:], value_states, 2)
    kv_states = past_key_value[self.layer_idx] if q_len == 1 else (key_states, value_states)
    key_states = kv_states[0]
    value_states = kv_states[1]
```

## 替换FlashAttention && cos/sin计算优化

**优化原因:** 将小算子替换为融合大算子，提升计算性能。

**FlashAttention优化方式：** 替换Qwen2VLAttention中q/k/v和attention_mask相关的两个矩阵乘，分别替换为
[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)和[torch_npu.npu_incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)。

```python
# Qwen2VLAttention类中新增函数_flash_attn，代替原来的_attn函数
def _flash_attn(self, q_len, query, key, value, attention_mask=None):
    attention_mask = attention_mask.to(torch.bool)
    if q_len > 1:
        attn_output = torch_npu.npu_prompt_flash_attention(query_states, key_states.contiguous(),
                                                        value_states.contiguous(), num_heads=self.num_heads,
                                                        input_layout="BNSD",
                                                        scale_value=self.scale_value,
                                                        pre_tokens=65535, next_tokens=65535,
                                                        atten_mask=attention_mask,
                                                        num_key_value_heads=self.num_key_value_heads)
    else:
        attn_output = torch_npu.npu_incre_flash_attention(query_states, key_states.contiguous(),
                                                        value_states.contiguous(), num_heads=self.num_heads,
                                                        input_layout="BNSD",
                                                        scale_value=self.scale_value,
                                                        atten_mask=attention_mask,
                                                        num_key_value_heads=self.num_key_value_heads)
    return attn_output


# Qwen2VLAttention::forward函数中调用小算子计算attention的地方替换
# 替换前：
# key_states = repeat_kv(key_states, self.num_key_value_groups)
# value_states = repeat_kv(value_states, self.num_key_value_groups)
# attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

# if attention_mask is not None:
#     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#     attn_weights = attn_weights + causal_mask

# if query_states.dtype == torch.float16:
#     attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

# attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
# attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
# attn_output = torch.matmul(attn_weights, value_states)

# 替换后：
attn_output = self._flash_attn(q_len, query, key, value, attention_mask)
```

**cos/sin计算优化方式：** 。
旋转位置编码替换为torch_npu.npu_apply_rotary_pos_emb融合算子。

```python
# 把cos和sin的计算移到Qwen2VLModel的forward中，并为Qwen2VLAttention新增参数cos和sin，传入Qwen2VLAttention
mrope_section = mrope_section * 2
cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    unsqueeze_dim
)
sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    unsqueeze_dim
)

# 替换torch_npu.npu_rotary_mul函数
#原代码
#q_embed = (q * cos) + (rotate_half(q) * sin)
#k_embed = (k * cos) + (rotate_half(k) * sin)
q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
```

## Add+RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能。

**优化方式**：替换RMSNorm的forward函数，使用融合算子[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_001031.html)和torch_npu.npu_add_rms_norm。

```python
def forward(self, hidden_states, residual: Optional[torch.Tensor]=None):
    if residual is None:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0], hidden_states
    else:
        y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
    return y, x
```

带来的其他地方的改动：在QWenBlock和Qwen2VLModel中都使用到了RMSNorm

```python
# Qwen2VLDecoderLayer::forward函数变动，先注释掉原先residual相关的代码
hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

# Self Attention
hidden_states, self_attn_weights, present_key_value = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=past_key_value,
    output_attentions=output_attentions,
    use_cache=use_cache,
    cache_position=cache_position,
    position_embeddings=position_embeddings,
    updated_kv_positions=updated_kv_positions,
    kv_padding_size=kv_padding_size,
    actual_seq_len=actual_seq_len,
    cos=cos,
    sin=sin,
)

# Fully Connected
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)

outputs = (residual, hidden_states,)

# Qwen2VLModel::forward函数变动
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
            cache_position,
            position_embeddings,
        )
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            past_residual=residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            updated_kv_positions=updated_kv_positions,
            kv_padding_size=kv_padding_size,
            actual_seq_len=actual_seq_len,
            cos=cos,
            sin=sin,
        )

    residual = layer_outputs[0]
    hidden_states = layer_outputs[1]

    if use_cache:
        next_decoder_cache = layer_outputs[3 if output_attentions else 2]

    if output_attentions:
        all_self_attns += (layer_outputs[2],)

hidden_states, _ = self.norm(hidden_states, residual)
```

## 全量优化计算量

优化原因：根据网络的计算逻辑，全量计算完logits后只取seq的最新位置的数据，因此在全量的最后线性层只需对最新的seq位置做计算，降低计算量

优化方式：

```python
# Qwen2VLForConditionalGeneration::forward函数计算logits的逻辑
# 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，所以在全量的最后线性层只需对最新的seq位置做计算
seq_len = hidden_states.shape[1]
if seq_len > 1:
    hidden_states = hidden_states[:, -1:, :]
lm_logits = self.lm_head(hidden_states)
```

## VIT使能PFA算子
**优化原因:** 将小算子替换为融合大算子，提升计算性能。

**优化方式：** 将VisualAttention中矩阵乘相关部分替换为
[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)。
```python
# 原代码
# q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
#         k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

# attention_mask = torch.full(
#     [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
# )
# for i in range(1, len(cu_seqlens)):
#     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

# q = q.transpose(0, 1)
# k = k.transpose(0, 1)
# v = v.transpose(0, 1)
# attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
# attn_weights = attn_weights + attention_mask
# attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
# attn_output = torch.matmul(attn_weights, v)
# attn_output = attn_output.transpose(0, 1)

# 替换代码
q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb)
k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb)
v = v.unsqueeze(0)

attention_mask = attention_mask.to(torch.bool)
attn_output = torch_npu.npu_prompt_flash_attention(q.contiguous(), k.contiguous(),
                                                    v.contiguous(), num_heads=self.num_heads,
                                                    input_layout="BSND",
                                                    scale_value=self.scale_value,
                                                    pre_tokens=65535, next_tokens=65535,
                                                    atten_mask=attention_mask, num_key_value_heads=self.num_heads)
```

## IFA算子计算优化+tiling全下沉

优化原因：IFA算子在进行增量计算时，由于输出做了padding，导致存在冗余的计算，可以通过传入actual_seq_lengths和kv_padding_size参数
来消除冗余的计算。同时使能tiling全下沉配置项，将IFA算子的tiling计算转移到AICPU侧异步执行，减少host-device间的同步和调度时延。

优化方式：
```python
# Step1：prepare_inputs_for_generation新增模型入参actual_seq_len和kv_padding_size
if input_ids.shape[1] == 1:
    # ifa Computational optimization inputs
    kv_padding_size = torch.tensor(self.config.max_position_embeddings - prompt_length,
                                   device=position_ids.device)
    actual_seq_len = (position_ids[:, -1] + 1).cpu().tolist()
else:
    kv_padding_size = None
    actual_seq_len = None


# Step2：使能IFA算子的actual_seq_lengths和kv_padding_size后，模型需要走动态模式，但模型实际输入为静态，
# 所以prepare_inputs_for_generation中需要对输入进行mark_static
if actual_seq_len is not None:
    self._mark_model_inputs_static(model_inputs)

def _mark_model_inputs_static(self, model_inputs):
    for key, value in model_inputs.items():
        if key == "past_key_values":
            for i in range(self.config.num_hidden_layers):
                torch._dynamo.mark_static(value[i][0])
                torch._dynamo.mark_static(value[i][1])
        elif key == "rotary_pos_emb":
            torch._dynamo.mark_static(value[0])
            torch._dynamo.mark_static(value[1])
        elif isinstance(value, torch.Tensor):
            torch._dynamo.mark_static(value)

            
# Step3：Qwen2VLAttention中torch_npu.npu_incre_flash_attention算子新增参数actual_seq_lengths和kv_padding_size：
attn_output = torch_npu.npu_incre_flash_attention(query, key.contiguous(),
                                                  value.contiguous(), num_heads=self.num_heads,
                                                  input_layout="BNSD",
                                                  scale_value=self.scale_value,
                                                  actual_seq_lengths=actual_seq_len,    # 新增
                                                  kv_padding_size=kv_padding_size,      # 新增
                                                  atten_mask=attention_mask)
```

**开启tiling全下沉配置，并将torch.compile的dynamic参数配置为True**
```python
# 在Qwen2VLForConditionalGeneration的check_execute_mode函数中新增配置项：
config.experimental_config.tiling_schedule_optimize = True
```

# 性能数据

## Qwen2-VL

执行Qwen2-VL，加载模型时的tensor类型bfloat16，输出max_new_tokens是128的性能数据如下：

**在910 B2的机器上，host是arm，1batch size，单卡性能数据：**

**该视频的分辨率为2048×1024，视频时长为5s，帧率为24帧/s**

**视频解码耗时：256ms (1725ms -> 256ms)**

**视频预处理耗时：20ms (70ms -> 20ms)**
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">优化项</th>
    <th class="tg-0pky" colspan="1">CPU</th>
    <th class="tg-0pky" colspan="2">NPU (DVPP / AIC)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">预处理总耗时</td>
    <td class="tg-0pky">70ms</td>
    <td class="tg-0pky">20ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">resize单算子</td>
    <td class="tg-0pky">10.8ms</td>
    <td class="tg-0pky">0.122ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">rescale单算子</td>
    <td class="tg-0pky">1.0ms</td>
    <td class="tg-0pky">0.117ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">normalize单算子</td>
    <td class="tg-0pky">3.3ms</td>
    <td class="tg-0pky">0.45ms</td>
  </tr>
</tbody>
</table>

**模型优化项：**
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">优化项</th>
    <th class="tg-0pky" colspan="2">单算子模式</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">Vit+全量</th>
    <th class="tg-0pky">增量</th>
    <th class="tg-0pky">Vit+全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">原始脚本</td>
    <td class="tg-0pky">147.5ms</td>
    <td class="tg-0pky">63.1ms</td>
    <td class="tg-0pky">138.5ms</td>
    <td class="tg-0pky">19.7ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Model替换FlashAttention</td>
    <td class="tg-0pky">124.7ms</td>
    <td class="tg-0pky">53.1ms</td>
    <td class="tg-0pky">131.8ms</td>
    <td class="tg-0pky">16.0ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">cos/sin优化</td>
    <td class="tg-0pky">117.9ms</td>
    <td class="tg-0pky">41.6ms</td>
    <td class="tg-0pky">115.0ms</td>
    <td class="tg-0pky">14.8ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">104.8ms</td>
    <td class="tg-0pky">31.6ms</td>
    <td class="tg-0pky">111.6ms</td>
    <td class="tg-0pky">14.6ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">104.1ms</td>
    <td class="tg-0pky">31.8ms</td>
    <td class="tg-0pky">109.3ms</td>
    <td class="tg-0pky">14.6ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Visual替换PFA</td>
    <td class="tg-0pky">97.4ms</td>
    <td class="tg-0pky">31.8ms</td>
    <td class="tg-0pky">91.6ms</td>
    <td class="tg-0pky">14.5ms</td>
  </tr>
</tbody>
</table>



# 性能测试

## 环境准备

**基于搭建的conda环境，安装配套版本的pytorch和torchvision，如torch==2.1.0和torchvision==0.16.0，并安装所需的python包**

```shell
# qwen-vl
pip3 install -r requirement.txt
```

**设置环境变量**

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/your/torchair/npu_tuned_model/mm/qwen-vl
cann_path=/usr/local/Ascend # 昇腾cann包安装目录
source ${cann_path}/latest/bin/setenv.bash
export ASCEND_HOME_PATH=${cann_path}/latest
```

## 单卡性能测试
**Step1：执行用例脚本**

```shell
# 图模式
python -u benchmark_qwen2_vl.py --model_path=xxx/Qwen2-VL-7B-Instruct
# 单算子
python -u benchmark_qwen2_vl.py --model_path=xxx/Qwen2-VL-7B-Instruct --execute_mode=eager
```

**性能数据打点位置**

参考图模式适配模型执行部分修改。全量耗时统计的是模型第一个step数据，增量耗时统计后续step数据的均值。