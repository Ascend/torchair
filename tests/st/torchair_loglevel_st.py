import unittest
import os
import logging
import torch
os.environ['TNG_LOG_LEVEL'] = '5'
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)

config = CompilerConfig()
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
        x = torch.randn(2, 2, 2)
        y = torch.randn(2, 2, 2)
        for i in range(2):
            model(x, y)

if __name__ == '__main__':
    unittest.main()