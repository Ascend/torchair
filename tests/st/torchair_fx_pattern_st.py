import math
import shutil
import contextlib
import os
import sys
import types
import logging
import unittest

import torch

try:
    import torch_npu
except ImportError:
    pass

import torchair
import logging
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
from torchair.core.utils import logger

torchair.logger.setLevel(logging.DEBUG)

config = torchair.CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = torchair.get_npu_backend(compiler_config=config)

### register npu device
from torchair_st_utils import capture_logger, generate_faked_module
import _privateuse1_backend

npu_device = _privateuse1_backend.npu_device()
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())


def stub_call(self, *args, **kwargs):
    return [torch.tensor(1.0), ]


### register npu custom ops
def scatter_update_meta(self, indices, updates, axis):
    return torch.empty_like(self)


def scatter_update__meta(self, indices, updates, axis):
    return self


lib = torch.library.Library("npu", "FRAGMENT")

if not hasattr(torch.ops.npu, "scatter_update"):
    lib.define("scatter_update(Tensor data, Tensor indices, Tensor updates, int axis) -> Tensor")
    torch.library.impl(lib, "scatter_update", "Meta")(scatter_update_meta)

if not hasattr(torch.ops.npu, "scatter_update_"):
    lib.define("scatter_update_(Tensor(a!) data, Tensor indices, Tensor updates, int axis) -> Tensor(a!)")
    torch.library.impl(lib, "scatter_update_", "Meta")(scatter_update__meta)


### register add_ converter
@torchair.register_fx_node_ge_converter(torch.ops.aten.add_.Tensor)
def conveter_aten_add_Tensor(self, other, *, alpha=1, meta_outputs=None):
    return torchair.ge.custom_op('Add',
                                 inputs={'x1': self, 'x2': other},
                                 outputs=['y'])


# define stub func
def stub_is_gte_cann_version(version, module="CANN"):
    logger.debug('[Stub] run stub func _is_gte_cann_version, and return True.')
    return True


def enable_remove_noop_ops_pattern(model, assert_func):
    def wrapper(*args, **kwargs):
        def wrapper_call(call):
            def wrapper(*args, **kwargs):
                ret = call(*args, **kwargs)
                assert_func(args[0])
                return ret

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        config_ = torchair.CompilerConfig()
        backend = torchair.get_npu_backend(compiler_config=config_)
        compiled_model = torch.compile(model, backend=backend, dynamic=True)
        _ = compiled_model(*args, **kwargs)

    return wrapper


# define stub utils submodule
class StubUtils:
    def __init__(self):
        logger.debug('[Stub] new stub module utils.')
        self._is_gte_cann_version = stub_is_gte_cann_version


# define stub npu submodule
class StubNpu:
    def __init__(self):
        logger.debug('[Stub] new stub module npu.')
        self.utils = StubUtils


def patch_torch_npu_module():
    original_module = None
    if 'torch_npu' in sys.modules:
        original_module = sys.modules['torch_npu']

    module = types.ModuleType('torch_npu_stub')
    module.npu = StubNpu
    module.__all__ = ['npu']

    sys.modules['torch_npu'] = module
    logger.debug('[Stub] Original torch_npu module is replaced by stub implementation: %s', sys.modules['torch_npu'])
    return original_module


def stub_is_supported_cann_version(target_version):
    logger.debug('[Stub] run stub cann version check of %s.', target_version)
    return True


class DsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, weight, smooth_scales):
        y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
        yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

        y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
        h1 = y1.size(-1)
        y2 = y1.view(-1, h1)
        yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)

        _, _, h2 = y1.shape
        y1 = y1.view(-1, h2).to(torch.float32)

        y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
        yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
        scale1Out3_view = scale1Out3.view(-1, 1)
        return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view


class TorchairFXPatternSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.call_bak = None
        self.original_torch_npu_module = None

    def setUp(self) -> None:
        self.call_bak = GeConcreteGraph.__call__
        return super().setUp()

    def tearDown(self) -> None:
        GeConcreteGraph.__call__ = self.call_bak
        return super().tearDown()

    def test_view_copy_not_inplace(self):
        from torchair.patterns import _recover_view_inplace_pattern
        _recover_view_inplace_pattern._is_supported_cann_version = stub_is_supported_cann_version

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data, indices, updates, axis=-2):
                view_1 = torch.ops.aten.view.default(data, [2, -1, 4, 16])
                view_2 = torch.ops.aten.view.default(updates, [2, -1, 1, 16])
                scatter_update_1 = torch.ops.npu.scatter_update.default(view_1, indices, view_2, axis=axis)
                view_3 = torch.ops.aten.view.default(scatter_update_1, [-1, 4, 16])
                view_4 = torch.ops.aten.view.default(view_3, [2, -1, 4, 16])
                add_1 = torch.ops.aten.add.Tensor(view_4, view_2)
                copy_ = torch.ops.aten.copy_(data, view_3)
                return add_1

        npu_config = torchair.CompilerConfig()
        npu_config.experimental_config.remove_noop_ops = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)

        data = torch.randn([2 * 8, 4, 16]).to(npu_device)
        updates = torch.randn([2 * 8, 1, 16]).to(npu_device)
        indices = torch.ones([2], dtype=torch.int64).to(npu_device)

        bak_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = stub_call
        with capture_logger() as stdout:
            res = model_dynamic(data, indices, updates)
        GeConcreteGraph.__call__ = bak_call
        captured_output = stdout.getvalue()
        self.assertTrue("In fx_graph, find 0 matched patterns" in captured_output)

    def test_view_add_inplace(self):
        from torchair.patterns import _recover_view_inplace_pattern
        _recover_view_inplace_pattern._is_supported_cann_version = stub_is_supported_cann_version

        from torchair.patterns._recover_view_inplace_pattern import _INPLACE_OPS_MAP, InplaceOpInfo
        _INPLACE_OPS_MAP.update({
            torch.ops.aten.add.Tensor: InplaceOpInfo(
                inplace_op=torch.ops.aten.add_.Tensor,
                output_to_ref_input_idx_mapping={0: 0})
        })

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data, updates):
                view_1 = torch.ops.aten.view.default(data, [2, -1, 4, 16])
                view_2 = torch.ops.aten.view.default(updates, [2, -1, 1, 16])
                scatter_update_1 = torch.ops.aten.add_.Tensor(view_1, view_2)
                view_3 = torch.ops.aten.view.default(scatter_update_1, [-1, 4, 16])
                view_4 = torch.ops.aten.view.default(view_3, [2, -1, 4, 16])
                add_1 = torch.ops.aten.sqrt.default(view_4)
                return add_1

        npu_config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)

        data = torch.randn([2 * 8, 4, 16]).to(npu_device)
        updates = torch.randn([2 * 8, 1, 16]).to(npu_device)

        bak_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = stub_call
        with capture_logger() as stdout:
            res = model_dynamic(data, updates)
        GeConcreteGraph.__call__ = bak_call
        captured_output = stdout.getvalue()
        self.assertTrue("success to replace non_inplace node add by inserting new inplace node add_" in captured_output)

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_disable_remove_noop_ops(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                slice_x = x[:]
                slice_y = y[:]
                return slice_x + slice_y

        def with_slice(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_slice_node = any(node.op == "call_function"
                                 and node.target.overloadpacket == torch.ops.aten.slice for node in nodes)
            self.assertTrue(has_slice_node)

        def wrapper_call(call, assert_):
            def wrapper(*args, **kwargs):
                ret = call(*args, **kwargs)
                self.assertGreater(len(args), 0)
                assert_(args[0])
                return ret

            return wrapper

        call_bak = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__, with_slice)
        try:
            model = Model()
            config_ = torchair.CompilerConfig()
            config_.experimental_config.remove_noop_ops = False
            backend = torchair.get_npu_backend(compiler_config=config_)
            compiled_model = torch.compile(model, backend=backend, dynamic=True)

            _ = compiled_model(torch.randn([2, 2]), torch.randn([2, 2]))
        finally:
            GeConcreteGraph.__call__ = call_bak

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_slice(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = x[:]
                y_ = y[:]
                return x_ + y_

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_slice_node = any(node.op == "call_function"
                                 and node.target.overloadpacket == torch.ops.aten.slice for node in nodes)
            self.assertFalse(has_slice_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_slice_scatter(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                slice_x = x.slice_scatter(y)
                return slice_x + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_slice_node = any(node.op == "call_function"
                                 and node.target.overloadpacket == torch.ops.aten.slice_scatter for node in nodes)
            self.assertFalse(has_slice_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_repeat(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                repeat_x = x.repeat(1, 1)
                return repeat_x + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_repeat_node = any(node.op == "call_function"
                                  and node.target.overloadpacket == torch.ops.aten.repeat for node in nodes)
            self.assertFalse(has_repeat_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_pad(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                import torch.nn.functional as func
                pad_x = func.pad(x, pad=[0, 0, 0, 0], value=3.5)
                return pad_x + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_pad_node = any(node.op == "call_function"
                               and node.target.overloadpacket == torch.ops.aten.constant_pad_nd for node in nodes)
            self.assertFalse(has_pad_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_convert(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                convert_x = torch.ops.prims.convert_element_type(x, torch.float32)
                return convert_x + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_convert_node = any(node.op == "call_function" and
                node.target.overloadpacket == torch.ops.prims.convert_element_type for node in nodes)
            self.assertFalse(has_convert_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_ceil(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = torch.ceil(x)
                return x_ + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.ceil for node in nodes)
            self.assertFalse(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.tensor([2, 2]), y=torch.tensor([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_pow(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = torch.pow(x, 1)
                return x_ + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.pow for node in nodes)
            self.assertFalse(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_cat(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = torch.cat([x], dim=0)
                return x_ + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.cat for node in nodes)
            self.assertFalse(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_clone(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = x.clone()
                return x_ + y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.clone for node in nodes)
            self.assertFalse(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_clone_skipped(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = x.clone()
                return x_

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.clone for node in nodes)
            self.assertTrue(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_clone_inplace(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x_ = x.clone()
                x_.add_(1)
                return y + x_

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_op_node = any(node.op == "call_function"
                              and node.target.overloadpacket == torch.ops.aten.clone for node in nodes)
            self.assertFalse(has_op_node)

        wrapped = enable_remove_noop_ops_pattern(Model(), assert_func)
        wrapped(x=torch.randn([2, 2]), y=torch.randn([2, 2]))

    def test_pattern_pass_for_aclgraph(self):
        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    def test_pattern_pass_for_ge(self):
        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    def test_close_pattern_pass_for_aclgraph(self):
        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_config.debug.aclgraph.enable_pattern_pass = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertFalse(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    def test_close_pattern_pass_for_ge(self):
        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.debug.aclgraph.enable_pattern_pass = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertFalse(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    def test_pattern_pass_for_aclgraph_with_multistream(self):
        class DsModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_event(tag="patternstream")

            def forward(self, x1, x2, weight, smooth_scales):
                def branch1():
                    y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                    yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                    y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return xOut, scale1Out, y1, xOut1, yOut

                def branch2(y1):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    h1 = y1.size(-1)
                    y2 = y1.view(-1, h1)
                    yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)
                    return yOut2, scale1Out2
                
                def branch3(y1):
                    _, _, h2 = y1.shape
                    y1 = y1.view(-1, h2).to(torch.float32)

                    y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                    yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
                    scale1Out3_view = scale1Out3.view(-1, 1)
                    return xOut3, yOut3, scale1Out3_view, y1
                
                with torch.npu.stream(self.stream1):
                    xOut, scale1Out, y1, xOut1, yOut = branch1()

                with torch.npu.stream(self.stream2):
                    yOut2, scale1Out2 = branch2(y1)

                with torch.npu.stream(self.stream1):
                    xOut3, yOut3, scale1Out3_view, y1 = branch3(y1)

                return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view
            
        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel2()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

if __name__ == '__main__':
    unittest.main()
