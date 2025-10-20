import math
import shutil
import contextlib
import os
import sys
import types
import logging
import unittest

import torch
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


if __name__ == '__main__':
    unittest.main()
