import unittest
import sys
from packaging import version

import torch

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair_st_utils import capture_logger

from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
    forbidden_attr,
    register_custom_ops,
)


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
        self.stub_module = StubNpu()
        register_custom_ops()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak
        return super().tearDown()

    def test_ge_eager_mode_dynamic_false(self):
        model = Model()
        config = CompilerConfig()
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

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
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

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
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

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
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
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

    def test_aclgraph_run_eagerly_with_static_kernel_compilation_range_check(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln = self.linear(x)
                return ln.mean()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "."
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = [5, 6]
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = TestModel()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)

        # first run, in valid range
        x0 = torch.randn([5, 2])
        with capture_logger() as stdout:
            model1(x0)
        self.assertTrue("Start to compile static shape kernel for fx graph" in stdout.getvalue())

        # run other shape, in valid range
        x1 = torch.randn([6, 2])
        with capture_logger() as stdout:
            model1(x1)
        self.assertTrue("Start to compile static shape kernel for fx graph" in stdout.getvalue())

        # run other shape, not in valid range
        x2 = torch.randn([7, 2])
        with capture_logger() as stdout:
            model1(x2)
        self.assertTrue("Skip compile static shape kernel for fx graph" in stdout.getvalue())

        # change the checked sym index
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_index = 2
        aclgraph_backend2 = torchair.get_npu_backend(compiler_config=config)

        model2 = TestModel()
        model2 = torch.compile(model2, backend=aclgraph_backend2, dynamic=True)

        # run second model, not in valid range
        x0 = torch.randn([5, 2])
        with capture_logger() as stdout:
            model2(x0)
        self.assertTrue("Skip compile static shape kernel for fx graph" in stdout.getvalue())

    def test_aclgraph_single_reinplace_within_mix_stream_scope(self):
        def test_func_same_stream(x, y, z):
            x2 = x - 1.0
            with torchair.scope.npu_stream_switch('o', 3):
                x.sub_(1.0)
            with torchair.scope.npu_stream_switch('1', 3):
                y.add_(1.0)
                y2 = y - 1.0
            with torchair.scope.npu_stream_switch('2', 3):
                z.mul_(1.0)
            return x2 + y2 + z

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("failed for reinplace. The users of the mutated input node have multiple streams."
                        in stdout.getvalue())
        self.assertTrue("success for reinplace. The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_multi_reinplace_within_same_stream_scope(self):
        def test_func_same_stream(x, cos, sin):
            x2 = x - 1.0
            with torchair.scope.npu_stream_switch('o', 3):
                x3 = x2 - sin  # 'o' stream
                ret = torch.ops.custom.sin_cos_inplace.default(x3, cos, sin)  # 'o' stream
                ret = cos + cos  # 'o' stream
            res = ret.sqrt()
            return res

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("success for reinplace. The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_multi_reinplace_within_mix_stream_scope(self):
        def test_func_same_stream(x, cos, sin):
            x2 = x - 1.0
            x3 = x2 - sin  # default stream
            with torchair.scope.npu_stream_switch('o', 3):
                ret = torch.ops.custom.sin_cos_inplace.default(x3, cos, sin)  # 'o' stream
                ret = ret * cos
            res = ret.sqrt()
            return res

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("failed for reinplace. The users of the mutated input node have multiple streams."
                        in stdout.getvalue())

    def test_aclgraph_single_reinplace_within_mix_stream_scope_success(self):
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('0', 3):
                x.sub_(1.0)
            with torchair.scope.npu_stream_switch('1', 3):
                y.add_(1.0)
                y2 = y - 1.0
            with torchair.scope.npu_stream_switch('2', 3):
                z.mul_(1.0)
            x.add_(3)
            return x + y2 + z

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertFalse("failed for reinplace. The users of the mutated input node have multiple streams."
                         in stdout.getvalue())
        self.assertTrue("success for reinplace. The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_multi_reinplace_within_mix_stream_scope_success(self):
        def test_func_same_stream(x, cos, sin):
            x.mul_(2)
            with torchair.scope.npu_stream_switch('o', 3):
                ret = torch.ops.custom.sin_cos_inplace.default(x, cos, sin)  # 'o' stream
                ret = cos + cos
            res = ret.sqrt()
            return res

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertFalse("failed for reinplace. The users of the mutated input node have multiple streams."
                         in stdout.getvalue())
        self.assertTrue("success for reinplace. The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_multi_reinplace_within_mix_stream_scope_fail(self):
        def test_func_same_stream(x, cos, sin):
            x.mul_(2)
            with torchair.scope.npu_stream_switch('o', 3):
                ret = torch.ops.custom.sin_cos_inplace.default(x, cos, sin)  # 'o' stream
                ret = ret * cos
            res = ret.sqrt()
            sin = cos + 1
            return res, sin

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3])
        y0 = torch.randn([3])
        z0 = torch.randn([3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("[multi_stream_multi_reinplace]Current node: sin_cos_functional, "
                        "type: custom.sin_cos_functional.default check no multi stream users failed for reinplace. "
                        "The users of the mutated input node have multiple streams. "
                        in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_reinplace_with_multi_stream_scope_success(self):
        # single_reinplace 非mutated_input，单流跨流能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x2 = x + 1.0
                # x2 has no user afterward 
                x3 = x2 + 2.0
                
            x4 = x3.view(-1)
            return x4

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[add_4] check reinplace is True, check multi-stream is True" in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_reinplace_with_multi_stream_scope_failed(self):
        # single_reinplace 非mutated_input，单流跨流不能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x2 = x + 1.0
                # x2 has user afterward
                x3 = x2 + 2.0
                
            x4 = x2 + 1.0
            x5 = x4 + x3
            return x5

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[add_4] check reinplace is False, check multi-stream is False" in stdout.getvalue())
        self.assertTrue("Node[add_8] check reinplace is True, check multi-stream is False" in stdout.getvalue())
        self.assertTrue("Node[add_12] check reinplace is True, check multi-stream is True" in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_reinplace_with_multi_streams_scope_success(self):
        # single_reinplace 非mutated_input，多流跨流能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x2 = x + 1.0
                # x2 has no user afterward
                x3 = x2 + 2.0
            
            with torchair.scope.npu_stream_switch('2', 3):
                x3 = x3 + 2.0
            x5 = x3.view(-1)
            return x5

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[add_4] check reinplace is True, check multi-stream is True" in stdout.getvalue())
        self.assertTrue("Node[add_8] check reinplace is True, check multi-stream is False" in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_reinplace_with_multi_streams_scope_failed_1(self):
        # single_reinplace 非mutated_input，多流跨流不能够替换（其他流）
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x2 = x + 1.0
                # x2 has user afterward
                x3 = x2 + 2.0
            
            with torchair.scope.npu_stream_switch('2', 3):
                x4 = x2 + 2.0
            x5 = x4.view(-1)
            return x5, x3

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[add_4] check reinplace is False, check multi-stream is False" in stdout.getvalue())
        self.assertTrue("Node[add_8] check reinplace is True, check multi-stream is False" in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_reinplace_with_multi_streams_scope_failed_2(self):
        # single_reinplace 非mutated_input，多流跨流不能够替换（默认流）
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x2 = x + 1.0
                # x2 has user afterward
                x3 = x2 + 2.0
            
            with torchair.scope.npu_stream_switch('2', 3):
                x4 = x3 + 2.0
            x5 = x2 + x3
            return x5, x4

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[add_4] check reinplace is False, check multi-stream is False" in stdout.getvalue())
        self.assertTrue("Node[add_8] check reinplace is False, check multi-stream is False" in stdout.getvalue())
        self.assertTrue("Node[add_12] check reinplace is True, check multi-stream is False" in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_mutated_reinplace_with_multi_stream_scope_success(self):
        # single_reinplace mutated_input，单流跨流能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                # x2 has user afterward
                x.add_(1.0)

            x5 = x.view(-1)
            return x5

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("[multi_stream_single_reinplace]Current node: add_3, "
                        "type: aten.add.Tensor "
                        "check no multi stream users success for reinplace. "
                        "The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_mutated_reinplace_with_multi_stream_scope_failed(self):
        # single_reinplace mutated_input，单流跨流不能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                # x2 has user afterward
                x.add_(1.0)

            x5 = x.mul_(2.0)
            return x5

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[mul_6] check reinplace is True, check multi-stream is True" in stdout.getvalue())
        self.assertTrue("[multi_stream_single_reinplace]Current node: mul_6, "
                        "type: aten.mul.Tensor "
                        "check no multi stream users success for reinplace. "
                        "The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())
        self.assertTrue("[multi_stream_single_reinplace]Current node: add_3, "
                        "type: aten.add.Tensor check no multi stream users failed for reinplace. "
                        "The users of the mutated input node have multiple streams. "
                        "All the users are [add_3, output]."
                        in stdout.getvalue())
        
    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_mutated_reinplace_with_multi_stream_scope_has_stream_label(self):
        # single_reinplace mutated_input，单流跨流不能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                # x2 has user afterward
                x.add_(1.0)

            x5 = x.mul_(2.0)
            x6 = x5 + 1
            return x6

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        # need add_stream_label_to_node_meta before reinplace
        config.experimental_config.pattern_fusion_pass = False
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[mul_6] check reinplace is True, check multi-stream is True" in stdout.getvalue())

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_aclgraph_single_mutated_reinplace_with_multi_streams_scope_success(self):
        # single_reinplace mutated_input，多流跨流能够替换
        def test_func_same_stream(x, y, z):
            with torchair.scope.npu_stream_switch('1', 3):
                x.add_(1.0)
            
            with torchair.scope.npu_stream_switch('2', 3):
                y.add_(x)
                y.mul_(2.0)
            x5 = x.view(-1)
            return x5

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = torch.compile(test_func_same_stream, backend=aclgraph_backend, dynamic=True)

        # first run
        x0 = torch.randn([3, 3])
        y0 = torch.randn([3, 3])
        z0 = torch.randn([3, 3])
        with capture_logger() as stdout:
            model1(x0, y0, z0)
        self.assertTrue("Node[mul_12] check reinplace is True, check multi-stream is True" in stdout.getvalue())
        self.assertTrue("[multi_stream_single_reinplace]Current node: add_3, "
                        "type: aten.add.Tensor "
                        "check no multi stream users success for reinplace. "
                        "The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())
        self.assertTrue("[multi_stream_single_reinplace]Current node: add_13, "
                        "type: aten.add.Tensor "
                        "check no multi stream users success for reinplace. "
                        "The users of the mutated input node did not have multiple streams."
                        in stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
