# npu\_fused\_infer\_attention\_score\_v2

## 支持的型号

<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

推理场景下，Ascend Extension for PyTorch提供的**torch\_npu.npu\_fused\_infer\_attention\_score\_v2**（参考《Ascend Extension for PyTorch 自定义 API参考》中的“torch\_npu.npu\_fused\_infer\_attention\_score”章节），适配增量和全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。

该接口在图模式场景下，如果开启[Tiling调度优化功能](Tiling调度优化功能.md)（config.experimental\_config.tiling\_schedule\_optimize），模型中actual\_seq\_length类参数会存在从Host到Device的拷贝开销，模型执行性能会下降。为此，**TorchAir提供了相应的定制化接口**，保障该算子Tiling调度优化效果。

本接口在Tiling下沉模式下，提供actual\_seq\_length类参数直接传Device Tensor的能力。原理是actual\_seq\_length类参数用于Tiling分核和Kernel计算，Tiling下沉时AI CPU中的Tiling分核和AI Core中的Kernel计算均在Device侧，直接传入Device可以减少Host到Device拷贝，从而降低开销。

## 函数原型

```
npu_fused_infer_attention_score_v2(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, int num_query_heads=1, int num_key_value_heads=1, float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout="BSH", int sparse_mode=0, int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None, int? out_dtype=None, ) -> (Tensor, Tensor)
```

## 参数说明

> **说明：** 
>-   actual\_seq\_length类参数：本接口是指actual\_seq\_lengths、actual\_seq\_lengths\_kv、actual\_shared\_prefix\_len参数。
>-   与torch\_npu.npu\_fused\_infer\_attention\_score\_v2接口相比，参数区别在actual\_seq\_length类参数类型支持Tensor，而非int型数组。


| 参数 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| 其他参数 | 输入 | 与torch_npu.npu_fused_infer_attention_score_v2接口同名参数要求一致。 | - |
| actual_seq_lengths | 输入 | Tensor类型，代表不同Batch中query的有效Sequence Length，数据类型支持int64。 | 否 |
| actual_seq_lengths_kv | 输入 | Tensor类型，代表不同Batch中key/value的有效Sequence Length，数据类型支持int64。 | 否 |
| actual_shared_prefix_len | 输入 | Tensor类型，代表key_shared_prefix/value_shared_prefix的有效Sequence Length。数据类型支持int64。 | 否 |

## 返回值说明

与torch\_npu.npu\_fused\_infer\_attention\_score\_v2接口返回值说明一致。

## 约束说明

-   本接口只支持图模式，**不支持Eager模式**下调用。
-   本接口仅适用于max-autotune模式。
-   其他约束与torch\_npu.npu\_fused\_infer\_attention\_score\_v2接口保持一致。

## 调用示例

```python
import torch
import torch_npu
import math
import torchair as tng

from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# 支持入图的打印宏
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)
from torch.library import Library, impl

# 数据生成
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
k_prefix = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v_prefix = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
actualSeqLengthkvs = [512]
actualSeqLengthkvs = torch.tensor(actualSeqLengthkvs).npu()
actualSeqLengths = [50]
actualSeqLengths = torch.tensor(actualSeqLengths).npu()
actualSeqLengthsPrefix = [50]
actualSeqLengthsPrefix = torch.tensor(actualSeqLengthsPrefix).npu()
scale = 1/math.sqrt(128.0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return tng.ops.npu_fused_infer_attention_score(q, k, v, actual_seq_lengths = actualSeqLengths, actual_seq_lengths_kv = actualSeqLengthkvs, key_shared_prefix = k_prefix, value_shared_prefix = v_prefix, actual_shared_prefix_len = actualSeqLengthsPrefix, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
def MetaInfershape():
    with torch.no_grad():
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        graph_output = model()
    print("graph output with mask:", graph_output[0], graph_output[0].shape)
if __name__ == "__main__":
    MetaInfershape()

# 执行上述代码的输出类似如下
graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
```

