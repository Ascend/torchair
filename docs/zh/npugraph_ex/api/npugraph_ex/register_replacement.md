# register\_replacement

## 功能说明

将自定义算子融合规则注册到npugraph\_ex中，在FX图编译后对图进行算子融合优化。

## 函数原型

```python
register_replacement(search_fn: SearchFn, replace_fn: ReplaceFn, example_inputs: Iterable[Any], trace_fn: TraceFn = fwd_only, extra_check: Callable[[Match], bool] = _return_true, search_fn_pattern: Union[PatternExpr, None] = None, scalar_workaround: Union[dict[str, Union[float, int]], None] = None, skip_duplicates: bool = False)
```

## 参数说明

|**参数**|**输入/输出**|**说明**|
|--|--|--|
|search_fn|输入|本函数作用是在FX图中识别算子组合或计算逻辑，如需要融合的算子组合。|
|replace_fn|输入|根据目标图中找到search_fn对应的组合后，使用本函数逻辑替换原有子图，实现算子融合或优化。|
|example_inputs|输入|用于追踪search_fn和replace_fn的示例输入张量。|
|trace_fn|输入|默认仅追踪前向计算图，适用于推理阶段的优化；若需支持训练场景，可传入支持反向追踪的函数。|
|extra_check|输入|对搜索到的算子组合进行额外校验的函数，函数的入参必须为torch._inductor.pattern_matcher中的Match对象，用于对匹配结果进行更多自定义判断，如判断算子组合是否在同一条流上/判断设备类型/判断入参形状等。|
|search_fn_pattern|输入|自定义的pattern对象，一般无需传入。定义参考PyTorch原生MultiOutputPattern对象的定义规则。传入该参数后，将不再使用search_fn来匹配算子组合，而是直接使用该参数作为匹配规则。|
|scalar_workaround|输入|用于显式绑定search_fn中标量参数值，用于匹配FX图追踪时固化的标量。|
|skip_duplicates|输入|用于控制注册阶段的重复检测行为。设为True时，若检测到重复的匹配模式，会跳过该重复模式的注册；设为False时，若检测到重复模式则直接报错，禁止该重复模式完成注册，该参数仅在PyTorch版本≥2.7.0时生效。|

## 返回值说明

无

## 约束说明

可通过[FX图算子融合Pass配置功能](../../basic/pattern_fusion_pass.md)将该接口功能关闭，默认开启。

## 调用示例

```python
import functools
import torch
from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode
import torch_npu

# 假设将add算子和npu_rms_norm算子融合成npu_add_rms_norm算子
# 定义一个search_fn, 用于查找原始FX图中融合之前的算子组合
def search_fn(x1, x2, gamma):
    x_out = torch.add(x1, x2)
    y, _ = torch_npu.npu_rms_norm(x_out, gamma)
    return y, x_out

# 定义一个replace_fn, 即融合算子，用于替换FX图中的算子组合
def replace_fn(x1, x2, gamma):
    y, _, x_out = torch_npu.npu_add_rms_norm(
        x1, x2, gamma
    )
    return y, x_out

# extra_check可以传入的额外校验逻辑，这里用于校验第一个输入参数x1的最后一维是否为特定值，
# 如果不是特定值则不允许融合
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
    input_tensor = functools.partial(torch.empty, (1, 1, 2), dtype=torch.float16, device="npu")
    kwargs_tensor = functools.partial(torch.empty, 2, dtype=torch.float16, device="npu")
    # 调用register_replacement接口，search_fn, replace_fn, example_inputs, 
    # 其中example_inputs为一个可迭代对象用来接受search_fn入参,
    # 如果有额外的校验，可传入extra_check
    torch.npu.npugraph_ex.register_replacement(
        search_fn=search_fn,
        replace_fn=replace_fn,
        example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
        extra_check=extra_check
    )

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, data1, data2, gamma):
        x_out = torch.add(data1, data2)
        y, _ = torch_npu.npu_rms_norm(x_out, gamma)
        abs_01 = torch.abs(y)
        sqrt_01 = torch.sqrt(x_out)
        return abs_01, sqrt_01

model = Model().npu()
x1 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
x2 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
gamma = torch.ones(7168, dtype=torch.float16, device='npu')
model_compile = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
res = model_compile(x1, x2, gamma)
```
