import os
os.environ['TNG_LOG_LEVEL'] = '0'
import time
import unittest
import torch
import torchair

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph import ge_apis as ge

import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
config.debug.graph_dump.type = "pbtxt"
npu_backend = torchair.get_npu_backend(compiler_config=config)


class TorchairSt(unittest.TestCase):
    def test_basic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
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

    def test_builtin_with_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                x = torch.add(x, y + z)
                x = torch.add(x, y - z)
                x = torch.add(x, y * z)
                x = torch.add(x, y / z)
                x = torch.add(x, y // z)
                return x

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x, 2, 3)
        model(x, 3, 4)

    def test_ge_api_support_position_passin_by_kv(self):
        # shape is position input of ge.Empty, check not raise when pass shape by k-v
        ge.Empty(shape=ge.Const(1))


if __name__ == '__main__':
    unittest.main()
