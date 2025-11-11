import unittest

import torch

import torchair
from torchair.configs.compiler_config import CompilerConfig


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)

    def forward(self, x):
        ln1 = self.linear1(x)
        ln2 = self.linear2(x)
        if x.sum() < 0:
            return ln1 + ln2 + 1
        else:
            return ln1 + ln2 - 1


class EagerModeSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_ge_eager_mode_dynamic_false(self):
        model = Model()
        config = CompilerConfig()
        config.debug.run_eagerly = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        x = torch.randn([3, 2])

        with torch.no_grad():
            original_output = model(x)

        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
        eager_output = compiled_model(x)
        self.assertTrue(torch.allclose(original_output, eager_output.cpu()))

    def test_ge_eager_mode_dynamic_true(self):
        model = Model()
        config = CompilerConfig()
        config.debug.run_eagerly = True
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        x = torch.randn([3, 2])

        with torch.no_grad():
            original_output = model(x)

        compiled_model = torch.compile(model, backend=npu_backend, dynamic=True)
        eager_output = compiled_model(x)
        self.assertTrue(torch.allclose(original_output, eager_output.cpu()))

    def test_acl_graph_eager_mode_dynamic_false(self):
        model = Model()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True

        # Acl graph do not support graph dump: config.debug.graph_dump.type = "py"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)

        x = torch.randn([3, 2])
        with torch.no_grad():
            original_output = model(x)
        compile_output = compiled_model(x)
        self.assertTrue(torch.allclose(original_output, compile_output.cpu()))

if __name__ == '__main__':
    unittest.main()
