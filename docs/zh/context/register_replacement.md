# register\_replacement

## 功能说明

将自定义算子融合规则注册到TorchAir框架中，在FX图编译后对图进行算子融合优化。

## 函数原型

```python
register_replacement(search_fn, replace_fn, example_inputs, trace_fn=fwd_only, extra_check=_return_true, search_fn_pattern=None)
```

## 参数说明


| 参数名 | 输入/输出 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| search_fn | 输入 | 该函数是希望在FX图中识别的算子组合或计算逻辑，如需要融合的算子组合 | 是 |
| replace_fn | 输入 | 在目标图中找到search_fn对应的组合时，会用这个函数的计算逻辑替换原有子图，实现算子融合或优化 | 是 |
| example_inputs | 输入 | 用于追踪search_fn和replace_fn的示例输入张量。输入的形状和dtype需与实际场景匹配。 | 是 |
| trace_fn | 输入 | 默认仅追踪前向计算图，适用于推理阶段的优化；若需支持训练场景，可传入支持反向追踪的函数。 | 否 |
| extra_check | 输入 | 找到算子组合后的额外校验函数，函数的入参必须为torch._inductor.pattern_matcher中的Match对象，用于对匹配结果进行更多自定义的判断，如判断算子组合是否在同一条流上/判断设备类型/判断入参形状等 | 否 |
| search_fn_pattern | 输入 | 自定义的pattern对象，一般无需传入。定义参考PyTorch原生MultiOutputPattern对象的定义规则。传入该参数后，将不再使用search_fn来匹配算子组合，而是直接使用该参数作为匹配规则。 | 否 |

## 返回值说明

无

## 约束说明

本接口适用于reduce-overhead和max-autotune模式。可通过[FX图算子融合Pass配置](FX图算子融合Pass配置.md)将该接口功能关闭，默认开启。

## 调用示例

简单示例如下，目标是将add算子和npu\_rms\_norm算子融合成npu\_add\_rms\_norm算子，并校验第一个输入参数的最后一维是否为特定值7168。

```python
import functools
import torch, torch_npu, torchair

from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode
from torchair.core.utils import logger

# 假设将add算子和npu_rms_norm算子融合成npu_add_rms_norm算子
# 定义一个search_fn, 用于查找原始FX图中融合之前的算子组合
def search_fn(x1, x2, gamma):
    xOut = torch.add(x1, x2)
    y, _ = torch_npu.npu_rms_norm(xOut, gamma)
    return y, xOut

# 定义一个replace_fn, 即融合算子，用于替换FX图中的算子组合
def replace_fn(x1, x2, gamma):
    y, _, xOut = torch_npu.npu_add_rms_norm(
        x1, x2, gamma
    )
    return y, xOut

# extra_check可以传入的额外校验逻辑，这里用于校验第一个输入参数x1的最后一维是否为特定值，如果不是特定值则不允许融合
def extra_check(match: Match):
    x1 = match.kwargs.get("x1")

    if x1 is None:
        return False 
    if not hasattr(x1, "meta") or "val" not in x1.meta:
        return False

    a_shape = x1.meta["val"].shape
    return a_shape[-1] == 7168 


# 定义一些样例输入，用于将search_fn和replace_fn追踪成FX图
fake_mode = FakeTensorMode()
with fake_mode:
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu", dtype=torch.float16)
    kwargs_tensor = functools.partial(torch.empty, 2, device="npu", dtype=torch.float16)

    # 调用torchair.register_replacement接口，search_fn, replace_fn, example_inputs, 如果有额外的校验，可传入extra_check
    torchair.register_replacement(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
        extra_check=extra_check
    )

# 正常调用torch.compile，执行torchair图下沉
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, data1, data2, gamma):
        xOut = torch.add(data1, data2)
        y, _ = torch_npu.npu_rms_norm(xOut, gamma)

        abs_01 = torch.abs(y)
        sqrt_01 = torch.sqrt(xOut)
        return abs_01, sqrt_01

npu_config = torchair.CompilerConfig()
npu_config.mode = "reduce-overhead"
npu_config.debug.graph_dump.type = "py"
npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
model = Model()

x1 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
x2 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
gamma = torch.ones(7168, dtype=torch.float16, device='npu')

model_compile = torch.compile(model, backend=npu_backend)
res = model_compile(x1, x2, gamma)

```

