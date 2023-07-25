import os
import time
import unittest
import torch
import torchair

import torchair
from torchair.configs.compiler_config import CompilerConfig

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"
npu_backend = torchair.get_npu_backend(compiler_config=config)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.add(x, y)


class TorchairSt(unittest.TestCase):
    def test_basic(self):
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(512, 1024, 1024)
        y = torch.randn(512, 1024, 1024)
        for i in range(2):
            model(x, y)

    def test_sym_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x, 2)
        model(x, 3)
        model(x, 2.0)
        model(x, 3.0)


if __name__ == '__main__':
    unittest.main()
