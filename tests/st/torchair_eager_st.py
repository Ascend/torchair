import unittest

import torch

import torchair
from torchair.configs.compiler_config import CompilerConfig

from torchair_st_utils import capture_logger


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

    def test_aclgraph_run_eagerly_with_static_kernel(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, x, y, z):
                ln1 = self.linear1(x)
                ln2 = self.linear2(y)
                return z + ln1.sum() + ln2.max()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "."
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = TestModel()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([6, 2])
        y0 = torch.randn([6, 3])
        z0 = torch.randn([8, 9])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Start to compile static shape kernel for fx graph" in stdout.getvalue())

        # second run with same shape
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Start to compile static shape kernel for fx graph" not in stdout.getvalue())

        # run other shape
        x1 = torch.randn([7, 2])
        y1 = torch.randn([7, 3])
        z1 = torch.randn([9, 11])
        with capture_logger() as stdout:
            model1(x1, y1, z1)
        self.assertTrue("Start to compile static shape kernel for fx graph" in stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
