# data\_dump

## 功能说明

GE图模式场景下，可通过本接口dump指定范围内的算子数据，并可以与[算子data dump功能](../../features/advanced/data_dump.md)其他dump options配套使用。

## 函数原型

```python
data_dump()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

- with语句块内不支持断图。

- 使用本接口时必须以with语句块形式调用，语句块内的算子信息均能被dump，具体参见下方调用示例。
- 本接口与dump layer配置项指定的算子范围均能生效，dump算子范围为两者并集，产物目录与dump layer一致。
- 本接口支持与上述所有dump配置项配合使用，产物目录基本一致。

## 调用示例

```python
import torch
import torchair
import logging
from torchair import logger
logger.setLevel(logging.DEBUG)

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, data0, data1):
        add_01 = torch.add(data0, data1)
        with torchair.scope.data_dump():
            sub_01 = torch.sub(data0, data1)
        return add_01, sub_01

input0 = torch.randn(2, 2, dtype=torch.float16).npu()
input1 = torch.randn(2, 2, dtype=torch.float16).npu()
config = torchair.CompilerConfig()
config.dump_config.enable_dump = True
config.dump_config.dump_layer = "Add"
npu_backend = torchair.get_npu_backend(compiler_config=config)
npu_mode = Network().npu()
npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
npu_out = npu_mode(input0, input1)
```
