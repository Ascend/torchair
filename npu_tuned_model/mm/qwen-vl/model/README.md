# Qwen-VL

本模块主要是Qwen-VL模型在npu上的适配迁移点以及部分性能优化项的介绍。
# 下载模型权重

模型介绍：https://github.com/QwenLM/Qwen-VL/tree/master

权重下载链接：https://huggingface.co/Qwen/Qwen-VL

# 图模式适配

[模型迁移指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/torchair/torchair_01_0001.html)

**注意**：需先保证模型在npu上的eager模式功能正常和精度正确，然后再进行以下图模式的迁移和适配。

为了方便后续图模式适配以及性能优化项的改造，需要先对模型结构进行调整，并把部分不支持入图的算子挪到模型的输入准备阶段：

```python
# ① 将VisionTransformer模块从QwenModel中挪到QWenLMHeadModel中，QWenLMHeadModel的__init__函数中添加：
self.visual = VisionTransformer(**config.visual)

# ② 将打开图片和图片预处理动作从visual.py中的encode函数挪到QWenLMHeadModel的prepare_inputs_for_generation函数中：
# 新增函数preprocess_images，并在prepare_inputs_for_generation中调用：
def preprocess_images(self, input_ids, past_key_values):
    if past_key_values is None and torch.any(input_ids == self.config.visual['image_start_id']):
        bos_pos = torch.where(input_ids == self.config.visual['image_start_id'])
        eos_pos = torch.where(input_ids == self.config.visual['image_start_id'] + 1)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        image_paths = []
        for i, a, b in img_pos:
            image = input_ids[i][a + 1: b - 1].tolist()
            image = image[: image.index(self.config.visual['image_start_id'] + 2)]
            image_paths.append(bytes(image).decode('utf-8'))

        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
    else:
        images = None
        img_pos = None
    return images, img_pos

# ③ 将rotary_pos_emb的计算从QwenModel中挪到QWenLMHeadModel的prepare_inputs_for_generation函数中：
# 新增函数compute_rotary_pos_emb，并在prepare_inputs_for_generation中调用：
def compute_rotary_pos_emb(self, input_ids, past_key_values):
    # past key values[0][0] shape: bs * seq_len * head_num * dim
    kv_seq_len = input_ids.size()[1] if past_key_values is None else past_key_values[0][0].shape[1]
    if (
            self.use_dynamic_ntk
            and kv_seq_len == input_ids.size()[1]
            and not self.training
    ):
        context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
    else:
        ntk_alpha = self.rotary_emb._ntk_alpha_cached
    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
    return rotary_pos_emb

# ④ 新增图模式接口，并在prepare_inputs_for_generation中调用：
def check_execute_mode(self):
    exe_mode = os.getenv("EXE_MODE", "dynamo")
    global compiled
    if exe_mode == "dynamo" and not compiled:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        npu_backend = tng.get_npu_backend(compiler_config=config)
        self.visual = torch.compile(self.visual, dynamic=False, fullgraph=True, backend=npu_backend)
        self.transformer = torch.compile(self.transformer, dynamic=True, fullgraph=True, backend=npu_backend)
        compiled = True
```

- torchair提供了NPU的图构造/图编译/图执行能力。相关能力全部集成到NPU图的后端，在使用torch.compile接口时，指定NPU图后端来使能；同时提供了开关和config控制图的编译和执行流程。
- 在使用NPU图后端的时候，torchair提供了静态图和动态图两种图执行的能力，根据dynamic参数决定是否走动态图。
- 由于模型结构发生了变化，需要对权重文件做对应的转换，避免权重加载报错：

```shell
# 权重转换
python3 convert_weight.py --model_path=xxx/Qwen-VL --output_path=xxx/Qwen-VL-Converted
```

# 性能优化

**注：** 在modeling_qwen.py和visual.py文件中，被修改的原函数都加了'_'前缀，可用于对比修改后的函数变化。下面列出了优化项主要的改动，不同的大模型适配完主要改动后，再根据模型实际代码逻辑进行调试。
如果期望使用的模型是Qwen-VL，可以直接使用改造后的modeling_qwen.py和visual.py。

## 固定kv cache大小

**优化原因** modeling_qwen.py的源码中kv cache作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种
更新方式存在多次申请内存及拷贝的性能损失。

优化方式：根据句子最大长度申请好一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv_cache进行更新。

```python
# modeling_qwen.py中QwenModel创建past_kv_values的逻辑挪到QWenLMHeadModel的prepare_inputs_for_generation函数中:
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
    past_key_values_length = self.config.max_position_embeddings if seq_length == 1 else 0
    if seq_length > 1:
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
        position_ids = torch.zeros(batch_size, dtype=position_ids.dtype, device=position_ids.device)
    else:
        bsz, src_len = attention_mask.size()
        padding_mask = torch.zeros(batch_size, self.config.max_position_embeddings, device=input_ids.device)
        padding_mask[:, :src_len] = attention_mask
        attention_mask = padding_mask
        position_ids = torch.ones(position_ids.shape, dtype=position_ids.dtype,
                                  device=position_ids.device) * (prompt_length - 1)

    attention_mask = self.transformer._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length),
                                                                      past_key_values[0][0], past_key_values_length)
    return past_key_values, attention_mask, position_ids
```

更新kv cache的改动：

```python
# 更新指定位置上的kv cache，position_ids在全量图执行时从起始位置更新，在增量图执行时从seq_len位置更新
# QWenAttention::forward函数中更新kv cache的逻辑改动如下：
tmp_ids = position_ids.reshape(-1)
torch_npu.scatter_update_(layer_past[0], tmp_ids, key, 1)
torch_npu.scatter_update_(layer_past[1], tmp_ids, value, 1)
key = layer_past[0] if q_len == 1 else key
value = layer_past[1] if q_len == 1 else value
present = layer_past if use_cache else None
```

## 替换FlashAttention && cos/sin计算优化

**优化原因:** 将小算子替换为融合大算子，提升计算性能。

**FlashAttention优化方式：** 替换QwenAttention中q/k/v和attention_mask相关的两个矩阵乘，分别替换为
[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)和[torch_npu.npu_incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)。

```python
# QWenAttention类中新增函数_flash_attn，代替原来的_attn函数
def _flash_attn(self, q_len, query, key, value, attention_mask=None):
    attention_mask = attention_mask.to(torch.bool)
    if q_len > 1:
        attn_output = torch_npu.npu_prompt_flash_attention(query.contiguous(), key.contiguous(),
                                                           value.contiguous(), num_heads=self.num_heads,
                                                           input_layout="BSND",
                                                           scale_value=self.scale_value,
                                                           pre_tokens=65535, next_tokens=0,
                                                           atten_mask=attention_mask)
    else:
        attn_output = torch_npu.npu_incre_flash_attention(query.contiguous(), key.contiguous(),
                                                          value.contiguous(), num_heads=self.num_heads,
                                                          input_layout="BSND",
                                                          scale_value=self.scale_value,
                                                          atten_mask=attention_mask)
    return attn_output


# QWenAttention::forward函数中调用_attn函数的地方改为调用_flash_attn
# 替换前：
#     query = query.permute(0, 2, 1, 3)
#     key = key.permute(0, 2, 1, 3)
#     value = value.permute(0, 2, 1, 3)
#     attn_output, attn_weight = self._attn(
#         query, key, value, registered_causal_mask, attention_mask, head_mask
#     )

# 替换后：
attn_output = self._flash_attn(q_len, query, key, value, attention_mask)
```

**cos/sin计算优化方式：** 。
旋转位置编码替换为torch_npu.npu_apply_rotary_pos_emb融合算子。

```python
# 替换apply_rotary_pos_emb函数
def apply_rotary_pos_emb(q, k, cos, sin):
    batch, seq_len, num_heads, head_dim = q.shape
    cos = cos.expand(batch, seq_len, 1, head_dim)
    sin = sin.expand(batch, seq_len, 1, head_dim)
    return torch_npu.npu_apply_rotary_pos_emb(q.contiguous(), k.contiguous(), cos, sin)

# QwenAttention::forward函数中使用计算好的cos、sin
if rotary_pos_emb is not None:
    query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb[0].to(value.dtype),
                                      rotary_pos_emb[1].to(value.dtype))
```

带来的其他改动点：

- 创建attention_mask时两处torch_full填充的值由min改为1

```python
# _make_causal_mask函数:
mask = torch.full((tgt_len, tgt_len), 1, device=device)
# _expand_mask函数:
inverted_mask.masked_fill(inverted_mask.to(torch.bool), 1).to(dtype)
```

## Add+RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能。

**优化方式**：替换RMSNorm的forward函数，使用融合算子[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_001031.html)和torch_npu.npu_add_rms_norm。

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

带来的其他地方的改动：在QWenBlock和QwenModel中都使用到了RMSNorm

```python
# QWenBlock::forward函数变动，先注释掉原先residual相关的代码
layernorm_output, residual = self.ln_1(hidden_states, past_residual)

attn_outputs = self.attn(
    layernorm_output,
    rotary_pos_emb,
    registered_causal_mask=registered_causal_mask,
    layer_past=layer_past,
    attention_mask=attention_mask,
    position_ids=position_ids,
    head_mask=head_mask,
    use_cache=use_cache,
    output_attentions=output_attentions,
)
attn_output = attn_outputs[0]

outputs = attn_outputs[1:]

layernorm_output, residual = self.ln_2(attn_output, residual)
hidden_states = self.mlp(layernorm_output)

if use_cache:
    outputs = (residual, hidden_states,) + outputs
else:
    outputs = (residual, hidden_states,) + outputs[1:]

# QwenModel::forward函数变动
residual = None
for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
        
    if self.gradient_checkpointing and self.training:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                # None for past_key_value
                return module(*inputs, use_cache, output_attentions)

            return custom_forward
        outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(block),
            hidden_states,
            rotary_pos_emb,
            self.registered_causal_mask,
            None,
            attention_mask,
            head_mask[i],
            encoder_hidden_states,
            encoder_attention_mask,
        )
    else:
        outputs = block(
            hidden_states,
            past_residual=residual,
            layer_past=layer_past,
            rotary_pos_emb=rotary_pos_emb,
            position_ids=position_ids,
            registered_causal_mask=self.registered_causal_mask,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
    residual = outputs[0]
    hidden_states = outputs[1]
    if use_cache is True:
        presents = presents + (outputs[2],)

    if output_attentions:
        all_self_attentions = all_self_attentions + (outputs[3 if use_cache else 2],)

hidden_states, _ = self.ln_f(hidden_states, residual)
```

## 全量优化计算量

优化原因：根据网络的计算逻辑，全量计算完logits后只取seq的最新位置的数据，因此在全量的最后线性层只需对最新的seq位置做计算，降低计算量

优化方式：

```python
# QWenLMHeadModel::forward函数计算logits的逻辑
# 由于logits最后也只取[:,-1,:]，相当于只取最新seq位置上的数据，所以在全量的最后线性层只需对最新的seq位置做计算
seq_len = hidden_states.shape[1]
if seq_len > 1:
    hidden_states = hidden_states[:, -1:, :]
lm_logits = self.lm_head(hidden_states)
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

            
# Step3：QwenAttention中torch_npu.npu_incre_flash_attention算子新增参数actual_seq_lengths和kv_padding_size：
attn_output = torch_npu.npu_incre_flash_attention(query, key.contiguous(),
                                                  value.contiguous(), num_heads=self.num_heads,
                                                  input_layout="BSND",
                                                  scale_value=self.scale_value,
                                                  actual_seq_lengths=actual_seq_len,    # 新增
                                                  kv_padding_size=kv_padding_size,      # 新增
                                                  atten_mask=attention_mask)
```

**开启tiling全下沉配置，并将torch.compile的dynamic参数配置为True**
```python
# 在QWenLMHeadModel的check_execute_mode函数中新增配置项：
config.experimental_config.tiling_schedule_optimize = True
```

## VIT离线计算位置编码

优化原因：当前visual.py中计算位置编码函数get_abs_pos中的interpolate算子性能较差，考虑到其输入参数都是已知的和输入图片无关，
可以提前将其结果离线计算好，从而减少模型执行的耗时。

优化方式：
```python
# Step1：将get_abs_pos的结果提前计算好并保存为模型的权重。在convert_weight.py文件中新增以下逻辑：
def get_abs_pos(x, tgt_size=1024):
    src_size = int(math.sqrt(x.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = x.dtype

    return F.interpolate(
        x.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2).contiguous(),
        size=(tgt_size, tgt_size),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


def replace_key(stat_dict, stat_dict_new):
    for key in stat_dict.keys():
        if "transformer.visual" in key:
            new_key = key.replace("transformer.visual", "visual")
        else:
            new_key = key

        # add if/elif to generate abs_pos_embed offline
        if "visual.positional_embedding" in new_key:
            stat_dict_new[new_key] = get_abs_pos(stat_dict[key])  # replace parameter with new value
        elif "visual.attn_pool.pos_embed" in new_key:
            stat_dict_new[new_key] = stat_dict[key]  # reserve origin parameter
            new_key = new_key.replace("pos_embed", "abs_pos_embed")
            stat_dict_new[new_key] = get_abs_pos(stat_dict[key])  # replace parameter with new value
        else:
            stat_dict_new[new_key] = stat_dict[key]


# Step2：将visual.py中调用get_abs_pos的地方替换为已经计算好的权重
# Resampler的forward函数，替换前：
'''
pos_embed = get_abs_pos(self.pos_embed, x.size(1))

out = self.attn(
    self._repeat(q, N) + self.pos_embed.unsqueeze(1),
    x + pos_embed.unsqueeze(1),
    x,
    attn_mask=attn_mask)[0]
'''
# 替换后：
out = self.attn(
    self._repeat(q, N) + self.pos_embed.unsqueeze(1),
    x + self.abs_pos_embed.unsqueeze(1),       # self.abs_pos_embed为提前计算好的权重
    x,
    attn_mask=attn_mask)[0]


# VisionTransformer的forward函数，替换前：
'''
x = x + get_abs_pos(self.positional_embedding, x.size(1))
'''

# 替换后：
x = x + self.positional_embedding     # self.positional_embedding为计算好的权重
```

## VIT使能FIA算子
**优化原因:** 将小算子替换为融合大算子，提升计算性能。

**优化方式：** 将VisualAttention中矩阵乘相关部分替换为：
[torch_npu.npu_fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000135.html)。
```python
def forward(self, query, key, value, attn_mask=None):
    # query/key/value: [b, sq, h]
    sq, b, _ = query.size()
    mixed_x_layer = self.in_proj(query)

    # [b, sq, (np * 3 * hn)] --> [b, sq, np, 3 * hn]
    new_tensor_shape = mixed_x_layer.size()[:-1] + \
                       (self.num_attention_heads_per_partition,
                        3 * self.hidden_size_per_attention_head)
    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

    # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
    query_layer, key_layer, value_layer = mixed_x_layer.split(
        self.hidden_size_per_attention_head, dim=-1)

    # [b, sq, np, hn] --> [b, np, sq, hn]
    query_layer = query_layer.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    value_layer = value_layer.transpose(1, 2)

    if 0 < self.res < 16:
        query_layer = F.pad(query_layer, [0, self.res])
        key_layer = F.pad(key_layer, [0, self.res])
        value_layer = F.pad(value_layer, [0, self.res])
    context_layer = torch_npu.npu_fused_infer_attention_score(query_layer, key_layer, value_layer,
                                                              num_heads=self.num_heads,
                                                              input_layout="BNSD_BSND",
                                                              scale=1. / self.norm_factor,
                                                              inner_precise=0)[0]
    if 0 < self.res < 16:
        context_layer = context_layer[..., :self.hidden_size_per_attention_head]

    context_layer = context_layer.reshape(b, sq, self.embed_dim)
    output = self.out_proj(context_layer)

    return output
```



# 性能数据

## Qwen-VL

执行Qwen-VL，加载模型时的tensor类型bfloat16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：

**在800I A2的机器上，host是arm，4batch size，单卡性能数据：**

**4张图片的分辨率分别为：2048×1365、1802×1309、1500×1001、6143×4095**

**图片预处理总耗时：938ms**


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
    <td class="tg-0pky">815ms</td>
    <td class="tg-0pky">71ms</td>
    <td class="tg-0pky">1149ms</td>
    <td class="tg-0pky">52ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">固定kv cache</td>
    <td class="tg-0pky">822ms</td>
    <td class="tg-0pky">65ms</td>
    <td class="tg-0pky">1035ms</td>
    <td class="tg-0pky">52ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">替换FlashAttention&amp;&amp;cos/sin优化</td>
    <td class="tg-0pky">612ms</td>
    <td class="tg-0pky">45ms</td>
    <td class="tg-0pky">888ms</td>
    <td class="tg-0pky">32ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">Add+RMSNorm融合</td>
    <td class="tg-0pky">587ms</td>
    <td class="tg-0pky">38ms</td>
    <td class="tg-0pky">849ms</td>
    <td class="tg-0pky">30ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">全量优化计算量</td>
    <td class="tg-0pky">558ms</td>
    <td class="tg-0pky">38ms</td>
    <td class="tg-0pky">844ms</td>
    <td class="tg-0pky">30ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">IFA算子计算优化+tiling全下沉</td>
    <td class="tg-0pky">557ms</td>
    <td class="tg-0pky">38ms</td>
    <td class="tg-0pky">887ms</td>
    <td class="tg-0pky">27ms</td>
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
    <th class="tg-0pky">Vit+全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">452ms</td>
    <td class="tg-0lax">23ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">2</td>
    <td class="tg-0pky">587ms</td>
    <td class="tg-0pky">25ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">4</td>
    <td class="tg-0lax">887ms</td>
    <td class="tg-0lax">27ms</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">1470ms</td>
    <td class="tg-0lax">30ms</td>
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
    <th class="tg-0pky">Vit+全量</th>
    <th class="tg-0pky">增量</th>
    <th class="tg-0pky">Vit+全量</th>
    <th class="tg-0pky">增量</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">VIT离线计算位置编码</td>
    <td class="tg-0pky">562ms</td>
    <td class="tg-0pky">38ms</td>
    <td class="tg-0pky">539ms</td>
    <td class="tg-0pky">26ms</td>
  </tr>
  <tr>
    <td class="tg-0pky">VIT使能FIA算子</td>
    <td class="tg-0pky">534ms</td>
    <td class="tg-0pky">38ms</td>
    <td class="tg-0pky">520ms</td>
    <td class="tg-0pky">26ms</td>
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
**Step0：转换模型权重**

```shell
python3 convert_weight.py --model_path=xxx/Qwen-VL --output_path=xxx/Qwen-VL-Converted
```

**Step1：执行用例脚本**

```shell
# 图模式
python -u benchmark_qwen_vl.py --model_path=xxx/Qwen-VL-Converted
# 单算子
python -u benchmark_qwen_vl.py --model_path=xxx/Qwen-VL-Converted --execute_mode=eager
```

**性能数据打点位置**

参考图模式适配模型执行部分修改。全量耗时统计的是模型第一个step数据，增量耗时统计后续step数据的均值。