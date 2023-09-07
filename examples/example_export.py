import os
from typing import Any, Dict, List, Tuple, Union
import functools
import logging
from torch._functorch.aot_autograd import aot_module_simplified
import torch

import torchair
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig

from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)
os.environ['TNG_LOG_LEVEL'] = '0'


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x + y
        z = torch.cat((x, y), 0)
        return z.size()[1], x


model = Model()
a = torch.randn(2, 4)
b = torch.randn(2, 4)


torchair.dynamo_export(a, b, model=model, export_path="./test_export_file_False", dynamic=False)
torchair.dynamo_export(a, b, model=model, export_path="./test_export_file_True", dynamic=True)
