# chatglm3-6B

本模块主要是chatglm3-6B模型在npu上的适配迁移点介绍。要求torch版本在2.x.x。由于原始模型CoreAttention forward函数中根据torch版本执行不同分支部分的处理逻辑在图模式下不支持，因此本模块只针对torch版本2.x.x分支进行图模式适配和优化。

# 下载模型权重及模型结构
chatglm3-6b https://huggingface.co/THUDM/chatglm3-6b

以下内容主要针对下载到本地的模型文件，即modeling_chatglm.py文件进行修改(本文基于commitid:f30825950ce00cb0577bf6a15e0d95de58e328dc的modeling_chtglm.py进行修改说明)。修改点主要包括torch_npu图模式适配，以及一些性能优化思路。

# 图模式适配

原始模型文件中存在attention_mask.all()算子调用，该算子无法入图，因此chatglm3图模式部分入图
```python
1.增加全局变量compiled标识只进行一次torch.compile的调用，初始值设置成False。
    compiled = False
2.将ChatGLMModel forward中部分代码封装成_wrap_graph函数，图模式下调用该函数，此处只进行了函数和封装，无其他改动。
    def _wrap_graph(self, position_ids, seq_length, inputs_embeds, full_attention_mask, past_key_values,
                    use_cache, output_hidden_states):
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )
        return hidden_states, presents, all_hidden_states, all_self_attentions

3.ChatGLMModel froward中根据模式不同调用wrap_graph函数
       if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        global compiled
        exe_mode = os.getenv("EXE_MODE", "eager")
        if exe_mode == "dynamo" and not compiled:
            logger.info("Start to run model in dynamo mode.")
            import torchair
            from torchair.config.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.experimental_config.frozen_parameter = True
            npu_backend = torchair.get_npu_backend(compiler_config = config)
            self._wrap_graph = torch.compile(self, dynamic=True, fullgraph=True,backend=npu_backend)
            compiled = True
        
        hidden_states, presents, all_hidden_states, all_self_attentions = self._wrap_graph(position_ids, seq_length,inputs_embeds, full_attention_mask,   past_key_values,use_cache,output_hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
4.替换部分
'''
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )
'''
```

# 性能优化

## 固定kv cache大小

**优化思路**：对于kv cache的处理是作为模型的输入，模型中通过cat进行拼接后，返回新的kv cache，这种更新方式存在多次申请内存及拷贝的性能损失。可以根据句子最大长度申请好一块固定大小的kv cache tensor，然后通过scatter_update_算子对指定位置上的kv cache进行更新。

**不适用原因**： torch_npu scatter_update_算子只支持后两个维度的更新，chatglm网络中seq_len在第0维度，如果使用npu算子还需要进行转置，增加时间开销，实测性能无提升，该优化点不适用于chatglm网络。



## 替换FlashAttention

**优化原因**：将小算子替换为融合大算子，提升性能

**FlashAttention优化方式**:替换CoreAttention中的Q/K/V和attention_mask相关的两个矩阵乘，分别替换为[torch_npu.npu_prompt_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000142.html)和[torch_npu.npu_incre_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000451.html)

```python
''' 替换部分
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
'''

# 替换后
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[3]:
            attention_mask = torch.ones(query_layer.size(0), 1, query_layer.size(2), key_layer.size(2),
                                        device = query_layer.device, dtype = torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if query_layer.ssize(2) > 1:
            attention_mask_pfa = attention_mask.to(torch.bool)
            context_layer = torch_npu.npu_prompt_flash_attention(query_layer.contiguous(),
                                                                 key_layer.contiguous(),
                                                                 value_layer.contiguous(),
                                                                 num_heads = query_layer.size(1),
                                                                 input_layout = "BNSD",
                                                                 scale_value = self.scale_value,
                                                                 pre_tokens = 65535,
                                                                 next_tokens = 65535,
                                                                 atten_mask = attention_mask_pfa)
        else:
            attention_mask_ifa = attention_mask.to(torch.bool)
            context_layer = torch_npu.npu_prompt_flash_attention(query_layer.contiguous(),
                                                                 key_layer.contiguous(),
                                                                 value_layer.contiguous(),
                                                                 num_heads = query_layer.size(1),
                                                                 input_layout = "BNSD",
                                                                 scale_value = self.scale_value,
                                                                 atten_mask = attention_mask_ifa)
```

## RMSNorm融合

**优化原因**：将小算子替换为融合大算子，提升性能

**优化方式**：替换RMSNorm的forward函数，使用融合算子[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_001031.html)和torch_npu.npu_rms_norm

```python
'''
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)
'''
class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, self.eps)[0]

```

## 全量优化计算量

**优化原因**：根据网络的计算逻辑，全量计算完logits后只取seq的最新位置的数据，所以在全量的最后线性层计算可以只对最新的seq位置做计算，降低计算量。原始模式已经存在该优化
```python
        if return_last_logit:
            hidden_states = hidden_states[-1:]
```


# 性能数据

## chatglm3

执行chatglm3-6b，加载模型时的tensor类型float16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：  
*优化点1：替换FlashAttention*	  
*优化点2：替换RMSNorm*  
**在800I A2的机器上，host是arm，4batch size：**

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">优化项</th>
    <th class="tg-0pky" colspan="2">单算子模式</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量(ms)</th>
    <th class="tg-0pky">增量(ms)</th>
    <th class="tg-0pky">全量(ms)</th>
    <th class="tg-0pky">增量(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">原始脚本</td>
    <td class="tg-0pky">489.7</td>
    <td class="tg-0pky">68.6</td>
    <td class="tg-0pky">447.7</td>
    <td class="tg-0pky">56.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">优化点(1)</td>
    <td class="tg-0pky">423.1</td>
    <td class="tg-0pky">61</td>
    <td class="tg-0pky">377.3</td>
    <td class="tg-0pky">48.8</td>
  </tr>
  <tr>
    <td class="tg-0pky">优化点(1,2)</td>
    <td class="tg-0pky">384.7</td>
    <td class="tg-0pky">53</td>
    <td class="tg-0pky">345.2</td>
    <td class="tg-0pky">44.7</td>
  </tr>
</tbody>
</table>


**800I A2在同时采用优化点1，2后不同bs的图模型性能比较：**
 tensor类型float16，输入padding到1024长度，输出max_new_tokens是1024的性能数据如下：
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">batch size</th>
    <th class="tg-0pky" colspan="2">图模式</th>
  </tr>
  <tr>
    <th class="tg-0pky">全量(ms)</th>
    <th class="tg-0pky">增量(ms)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">1</td>
    <td class="tg-0lax">108.3</td>
    <td class="tg-0lax">27.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">345.2</td>
    <td class="tg-0pky">44.7</td>
  </tr>
  <tr>
    <td class="tg-0lax">8</td>
    <td class="tg-0lax">758.2</td>
    <td class="tg-0lax">69.7</td>
  </tr>
  <tr>
    <td class="tg-0lax">16</td>
    <td class="tg-0lax">1474.5</td>
    <td class="tg-0lax">117.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">32</td>
    <td class="tg-0lax">2937.4</td>
    <td class="tg-0lax">220.8</td>
  </tr>
  <tr>
    <td class="tg-0pky">48</td>
    <td class="tg-0pky">4420.9</td>
    <td class="tg-0pky">312.3</td>
  </tr>
</tbody>
</table>

# 性能测试
benchmark_chatglm.py脚本提供了对接deepspeed多卡切分执行chatglm的样例参考, 本文性能数据基于deepspeed（0.14.1） transformers(4.31.0)在arm host + 800I A2环境执行进行统计   
下载模型权重及模型文件 
```bash  
git lfs install  
git clone git@hf.co:THUDM/chatglm3-6b chatglm3  
git checkout f30825950ce00cb0577bf6a15e0d95de58e328dc  
```
将当前目录中的modeling_chatglm.py文件替换下载目录下的modeling_chatglm.py文件  
```bash
cp modeling_chatglm.py chatglm3/modeling_chatglm.py
```
设置环境变量 
```bash
export PYTHONPATH=$PYTHONPATH:chatglm3/
```
deepspeed 拉起chatglm模型在8卡上切分执行
```bash
deepspeed --num_gpus=8 benchmark_chatglm.py --model_path=chatglm3/ --execute_mode="dynamo"
```
1. promts维度对应batch size，样例中为4
2. main函数中input_max_len指定seq_len
3. model_runner.execute_mode = "dynamo"指明脚本默认执行图模式，如果需要执行单算子模式需要将该属性设置为eager
4. jit_compile=False表示在图模式下开启二进制编译，提高图模式下编译阶段性能。

