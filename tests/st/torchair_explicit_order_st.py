import contextlib
import functools
import threading
import unittest
import sys

import torch
import _privateuse1_backend
import torchair
from torchair._ge_concrete_graph.graph_pass import explict_order_for_side_effect_nodes
from torchair._ge_concrete_graph import graph_pass


captured_graph = threading.local()
_privateuse1_backend.register_hook()


def _get_op_inputs(graph):
    op_inputs = dict()
    for op in graph.op:
        op_inputs[op.name] = [v for v in op.input]
    return op_inputs


@functools.wraps(explict_order_for_side_effect_nodes)
def capture_func(graph, *args, **kwargs):
    captured_graph.origin_op_inputs = _get_op_inputs(graph)
    captured_graph.graph = graph
    return explict_order_for_side_effect_nodes(graph, *args, **kwargs)


@contextlib.contextmanager
def capture_ge_graph():
    try:
        from torchair.ge._ge_graph import _g_name_dict
        _g_name_dict.clear()
        captured_graph.origin_op_inputs = None
        captured_graph.graph = None
        yield captured_graph
    finally:
        pass


graph_pass.explict_order_for_side_effect_nodes = capture_func


class FakeTorchNpu:

    def __getattr__(self, item):
        return self

    @staticmethod
    def get_npu_format(*args, **kwargs):
        return 0

    @staticmethod
    def current_device():
        return 0


_initialized = False


def _mock_npu():
    global _initialized
    if _initialized:
        return
    from torchair.core import _npu_graph_executor
    torch.utils.rename_privateuse1_backend("npu")
    _initialized = True


@contextlib.contextmanager
def npu_ctx():
    _mock_npu()
    origin_torch_npu = sys.modules.get('torch_npu', None)
    try:
        sys.modules['torch_npu'] = FakeTorchNpu()
        yield
    finally:
        if origin_torch_npu:
            sys.modules['torch_npu'] = origin_torch_npu


def npu_tensor(*args, **kwargs):
    _mock_npu()
    t = torch.ones(*args, **kwargs)
    t.is_npu = True
    return t


class NpuExplictOrderSt(unittest.TestCase):
    def test_unchanged(self):
        def func(v):
            return torch.add(v, 1)

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor((2, 2), device='npu'))

        self.assertEqual(captured.origin_op_inputs, _get_op_inputs(captured.graph))

    def test_not_pruned(self):
        def func(v):
            torchair.ops.npu_print(v)

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor(2, device='npu'))

        origin_op_inputs = captured.origin_op_inputs
        current_op_inputs = _get_op_inputs(captured.graph)

        self.assertEqual(origin_op_inputs["NetOutput"], current_op_inputs["NetOutput"][:-1])
        self.assertEqual(current_op_inputs["NetOutput"][-1], "PrintV2:-1")

    def test_multi_stateful(self):
        def func(v):
            torchair.ops.npu_print(v)
            torchair.ops.npu_print(v)
            torchair.ops.npu_print(v)

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor(2, device='npu'))

        origin_op_inputs = captured.origin_op_inputs
        current_op_inputs = _get_op_inputs(captured.graph)

        self.assertEqual(origin_op_inputs["NetOutput"], current_op_inputs["NetOutput"][:-1])
        self.assertEqual(current_op_inputs["NetOutput"][-1], "PrintV2_2:-1")
        self.assertEqual(current_op_inputs["PrintV2_1"][-1], "PrintV2:-1")
        self.assertEqual(current_op_inputs["PrintV2_2"][-1], "PrintV2_1:-1")

    def test_multi_stateful_with_inplace_1(self):
        def func(v):
            torchair.ops.npu_print(v)
            v.add_(1)
            torchair.ops.npu_print(v)

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor((2, 2), device='npu'))

        origin_op_inputs = captured.origin_op_inputs
        current_op_inputs = _get_op_inputs(captured.graph)

        self.assertEqual(origin_op_inputs["NetOutput"], current_op_inputs["NetOutput"][:-1])
        self.assertEqual(current_op_inputs["NetOutput"][-1], "PrintV2_1:-1")
        self.assertEqual(current_op_inputs["Add"][-1], "PrintV2:-1")
        self.assertEqual(current_op_inputs["PrintV2_1"][-1], "PrintV2:-1")

    def test_multi_stateful_with_inplace_2(self):
        def func(v):
            v.add_(1)
            torchair.ops.npu_print(v)
            v.add_(1)

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor((2, 2), device='npu'))

        origin_op_inputs = captured.origin_op_inputs
        current_op_inputs = _get_op_inputs(captured.graph)

        self.assertEqual(origin_op_inputs["NetOutput"], current_op_inputs["NetOutput"][:-1])
        self.assertEqual(current_op_inputs["NetOutput"][-1], "PrintV2:-1")

    def test_unchanged_when_no_side_effects(self):
        def func(v):
            x = torch.add(v, 1)
            v.add_(1)
            return x

        with npu_ctx(), capture_ge_graph() as captured:
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(npu_tensor((2, 2), device='npu'))

        self.assertEqual(captured.origin_op_inputs, _get_op_inputs(captured.graph))


if __name__ == '__main__':
    unittest.main()
