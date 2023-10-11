
from typing import Any, Dict, List, Tuple, Union
from torch._functorch.aot_autograd import aot_module_simplified
import torch
import functools
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
# dump开关：[必选] 开启dump
config.dump_config.enable_dump = True
# dump类型：[可选][input、output、all]分别代表dump输入、输出、所有数据，默认为dump所有数据
config.dump_config.dump_mode = 'all'
# dump路径：[可选]默认为当前目录
config.dump_config.dump_path = './dump'
npu_backend = tng.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)


model = Model()
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(2, 2)
y = torch.randn(2, 2)
model(x, y)


# 修改config后，与第一次设置的不一样，会校验报错
config.dump_config.enable_dump = True
config.dump_config.dump_mode = 'input'
config.dump_config.dump_path = './dump_other'
torch._dynamo.reset()
npu_backend = tng.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=True)
x = torch.randn(2, 2)
y = torch.randn(2, 2)
model(x, y)