import contextlib
import dataclasses
import functools
import logging
import os
import sys
import types
import unittest
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
import torchair
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig

logger.setLevel(logging.DEBUG)

"""Start to gen some API patch for AclGraph in st."""


# define stub FA API
def stub_npu_fa_func(*args, **kwargs):
    logger.debug('[Stub] using stub implementation of NPU FA with args: %s and kwargs: %s', args, kwargs)
    return torch.randn([3, 2])
    return torch.empty_like(args[0])  # 示例实现


class StubNpuFA:
    def __init__(self):
        pass


stub_fa = StubNpuFA()
stub_fa.default = stub_npu_fa_func
stub_fa.out = stub_npu_fa_func


# define stub aclgraph API
def stub_graph_pool_handle():
    logger.debug('[Stub] run stub API graph_pool_handle with args[].')
    pass


def stub_synchronize():
    logger.debug('[Stub] run stub API stream synchronize with args[].')
    pass


class StubNPUGraph:
    def __init__(self):
        logger.debug('[Stub] new stub class NPUGraph.')
        pass

    def replay(self):
        logger.debug('[Stub] run stub API replay with args[].')
        pass


class graph:
    def __init__(
            self,
            npu_graph,
            pool=None,
            stream=None,
            capture_error_mode: str = "global"):
        logger.debug('[Stub] new stub class graph with args[%s, %s, %s, %s].',
                     type(npu_graph), pool, stream, capture_error_mode)
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class StubStream:
    def __new__(cls, device=None, priority=0, **kwargs):
        logger.debug('[Stub] new stub class Stream.')
        return "stream"

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        pass


def current_stream(device=None):
    logger.debug('[Stub] run stub API current_stream with args[].')
    return "current_stream"


# define stub submodule
class StubNpu:
    def __init__(self):
        logger.debug('[Stub] new stub module npu.')
        self.npu_fused_infer_attention_score = stub_fa
        self._npu_fused_infer_attention_score_get_max_workspace = stub_fa
        self.NPUGraph = StubNPUGraph
        self.graph = graph
        self.Stream = StubStream
        self.current_stream = current_stream
        self.graph_pool_handle = stub_graph_pool_handle
        self.synchronize = stub_synchronize


def patch_ops_npu_module(stub_module):
    original_module = None
    original_exists = hasattr(torch.ops, 'npu')
    if original_exists:
        original_module = torch.ops.npu

    torch.ops.npu = stub_module
    logger.debug('[Stub] Original torch.ops.npu module is replaced by stub implementation: %s', torch.ops.npu)
    return original_module


def patch_torch_point_npu_module(stub_module):
    original_module = None
    original_exists = hasattr(torch, 'npu')
    if original_exists:
        original_module = torch.npu

    torch.npu = stub_module
    logger.debug('[Stub] Original torch.npu module is replaced by stub implementation: %s', torch.npu)
    return original_module


def patch_torch_npu_module(stub_module):
    original_module = None
    if 'torch_npu' in sys.modules:
        original_module = sys.modules['torch_npu']

    module = types.ModuleType('torch_npu_stub')
    module.npu = stub_module
    module.__all__ = ['npu']

    sys.modules['torch_npu'] = module
    logger.debug('[Stub] Original torch_npu.npu module is replaced by stub implementation: %s',
                 sys.modules['torch_npu'])
    return original_module


class AclGraphSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        return super().tearDown()

    def test_aclgraph_capture_and_replay(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        # config.debug.graph_dump.type = "pbtxt"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)
        x = torch.randn([3, 2])
        for i in range(2):
            model(x)

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.mul_(2)
                return x + 1

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # inference
        with self.assertLogs(logger, level="WARNING") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("data_ptr is different between capture and replay." in log for log in cm.output),
            f"Expected WARNING 'Mutated input[arg]'s data_ptr is different between capture and replay.' "
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_false(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.div_(2)
                return x - 1

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # expected no warning called
        from unittest.mock import patch
        with patch("logging.Logger.warning") as mock_warning:
            for _ in range(2):
                output = model(x)
            mock_warning.assert_not_called()

    def test_aclgraph_update(self):
        from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
        _REPLACE_FUNC_MAP[torch.ops.aten.max_unpool2d.default] = StaticWorkspaceReplaceFunc(
            get_workspace=None,
            out_operator=torch.ops.aten.max_unpool2d.out,
            workspace_keys=[],
            output_keys=["out"],
            updated_param_keys=[],
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, output_size):
                val = torch.nn.functional.max_unpool2d(x, y, output_size)
                return val.mean()

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        # config.debug.graph_dump.type = "pbtxt"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        output, indices = F.max_pool1d(
            torch.randn([1, 1, 4]), 2, stride=2, return_indices=True
        )

        torch._dynamo.mark_static(output)
        torch._dynamo.mark_static(indices)
        with self.assertRaisesRegex(RuntimeError, r'which is not in updated index'):
            model(output, indices, 2)

    def test_aclgraph_unsupported_dynamic_sym_in_tensor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])
        with self.assertRaisesRegex(RuntimeError, r'with sym in graph input tensor'):
            model(x)

    def test_aclgraph_unsupported_dump(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x - 1.0

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.graph_dump.type = "pbtxt"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)
        x = torch.randn([3, 2])
        with self.assertRaisesRegex(RuntimeError, r'for acl graph is not implemented!'):
            model(x)


if __name__ == '__main__':
    unittest.main()
