import os
import sys

os.environ['TNG_LOG_LEVEL'] = '0'
import unittest
import torch

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


def register_npu():
    import torch_npu.npu as npu_device
    npu_device.register_npu()


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

    def test_auto_tune(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=False)
        x = torch.randn(512, 1024, 1024)
        y = torch.randn(512, 1024, 1024)
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

    def test_auto_tune(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=False)
        x = torch.randn(2, 2)
        model(x, 2)

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

    def test_dynamic_npu_graph_executor_error(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        # 注册npu
        register_npu()
        x = torch.randn(512, 1024, 1024)
        y = torch.randn(512, 1024, 1024)
        model(x, y)

    def test_dynamic_npu_graph_executor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        # 注册npu
        register_npu()
        device = 'privateuseone' if ('torch_npu' in sys.modules) else 'cpu'
        x = torch.randn(512, 1024, 1024).to(device)
        y = torch.randn(512, 1024, 1024).to(device)
        model(x, y)
        model(x, y)

    def test_static_npu_graph_executor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        # 注册npu
        register_npu()
        device = 'privateuseone' if ('torch_npu' in sys.modules) else 'cpu'
        x = torch.randn(512, 1024, 1024).to(device)
        y = torch.randn(512, 1024, 1024).to(device)
        model(x, y)
        model(x, y)

    def test_npu_executor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        import torch_npu.npu
        device = 'privateuseone' if ('torch_npu' in sys.modules) else 'cpu'
        x = torch.randn(512, 1024, 1024).to(device)
        y = torch.randn(512, 1024, 1024).to(device)
        model(x, y)
    
    def test_ge_graph_dump_with_py(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        def get_dumped_py_file_list(dir_path, file_extension='.py'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo_') and i.endswith(f'{file_extension}')]

        for file_name in get_dumped_py_file_list('./'):
            os.remove(os.path.join('./', file_name))

        config = CompilerConfig()
        config.debug.graph_dump.type = "py"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        model = Model()
        model = torch.compile(model, backend=npu_backend)
        x = torch.randn(2, 2)
        output = model(x)

        dumped_py_file_list = get_dumped_py_file_list('./')
        dumped_py_file_list.sort(key=lambda file_name: os.path.getmtime(os.path.join('./', file_name)))
        assert dumped_py_file_list.__len__() > 0
        file_name = os.path.join('./', dumped_py_file_list[-1])

        with open(file_name, 'r')as f:
            src = f.read()

        assert src != '# -*- coding: utf-8 -*-\nfrom torch import tensor\n' \
                      'from torchair.ge_concrete_graph import ge_apis as ge\n' \
                      'from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph\n\n'

        exec(src)

if __name__ == '__main__':
    unittest.main()
