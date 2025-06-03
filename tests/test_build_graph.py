import os
import logging
import unittest
from typing import Set
from contextlib import contextmanager

from npu_extension_for_inductor.npu import NPUKernel
import npu_extension_for_inductor
from torch._inductor.virtualized import V
import torch

logging.basicConfig(level=logging.INFO)


@contextmanager
def test_with_env(**kwargs):
    old_env = {}
    for key, value in kwargs.items():
        old_env[key] = os.getenv(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


class KernelCapture:
    def __init__(self):
        self.kernels: Set[NPUKernel] = set()
        self.origin = V.set_kernel_handler

    def watch_npu_kernel(self, kernel):
        if isinstance(kernel, NPUKernel):
            self.kernels.add(kernel)
        return self.origin(kernel)

    def kernel(self, index):
        return sorted(self.kernels, key=lambda k: int(k.kernel_name.split('_')[0][3:]))[index]

    def graph(self, index):
        return self.kernel(index).fused_graph

    def graph_str(self, index, replace_name):
        graph = self.graph(index)
        return graph.codegen(replace_name).getvalue()

    def __enter__(self):
        self.kernels.clear()
        self.origin = V.set_kernel_handler
        V.set_kernel_handler = self.watch_npu_kernel
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        V.set_kernel_handler = self.origin


class BuildGraphTest(unittest.TestCase):
    def assert_graph_equal(self, actual, expect):
        def str_to_graph_lines(s: str):
            lines = [line.strip() for line in s.split('\n') if line.strip()]
            for start, line in enumerate(lines):
                if "ascir.HintGraph" in line:
                    return lines[start:]
            assert False, "Can't find graph definition"

        actual = str_to_graph_lines(actual)
        expect = str_to_graph_lines(expect)
        for i in range(len(actual)):
            if actual[i] != expect[i]:
                self.assertEqual(actual[i].replace('TrueDiv', 'Div').replace('truediv', 'div'),
                                 expect[i].replace('TrueDiv', 'Div').replace('truediv', 'div'))

    def test_softmax_graph(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        with KernelCapture() as kernel_capture:
            x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
            test_softmax(x)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "fused_graph"),
                                f"""# --------------------graph0--------------------
graph0_hint = ascir.HintGraph('graph0_hint')
s0 = graph0_hint.create_size("s0")
s1 = graph0_hint.create_size("s1")
s2 = graph0_hint.create_size("s2")
z0 = graph0_hint.create_axis("z0", s0*s1)
z1 = graph0_hint.create_axis("z1", s2)
data = ascir.ops.Data('graph0_hint/data', graph0_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float16
load = ascir.ops.Load('graph0_hint/load', graph0_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
cast = ascir.ops.Cast('graph0_hint/cast', graph0_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = load.y
cast.y.dtype = ascir.dtypes.float32
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
max = ascir.ops.Max('graph0_hint/max', graph0_hint)
max.attr.sched.axis = [z0, z1]
max.x = cast.y
max.y.axis = [z0, z1]
max.y.size = [s0*s1, 1]
max.y.strides = [1, 0]
store = ascir.ops.Store('graph0_hint/store', graph0_hint)
store.attr.sched.axis = [z0, z1]
store.x = max.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, 1]
store.y.strides = [1, 0]
output = ascir.ops.Output('graph0_hint/output', graph0_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph0_hint.infer_dtypes()
# --------------------graph1--------------------
graph1_hint = ascir.HintGraph('graph1_hint')
s0 = graph1_hint.create_size("s0")
s1 = graph1_hint.create_size("s1")
s2 = graph1_hint.create_size("s2")
z0 = graph1_hint.create_axis("z0", s0*s1)
z1 = graph1_hint.create_axis("z1", s2)
data = ascir.ops.Data('graph1_hint/data', graph1_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float16
load = ascir.ops.Load('graph1_hint/load', graph1_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
cast = ascir.ops.Cast('graph1_hint/cast', graph1_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = load.y
cast.y.dtype = ascir.dtypes.float32
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
data1 = ascir.ops.Data('graph1_hint/data1', graph1_hint)
data1.attr.ir_attr.index = 1
data1.y.dtype = ascir.dtypes.float32
load1 = ascir.ops.Load('graph1_hint/load1', graph1_hint)
load1.attr.sched.axis = [z0, z1]
load1.attr.ir_attr.offset = 0
load1.x = data1.y
load1.y.axis = [z0, z1]
load1.y.size = [s0*s1, 1]
load1.y.strides = [1, 0]
broadcast = ascir.ops.Broadcast('graph1_hint/broadcast', graph1_hint)
broadcast.attr.sched.axis = [z0, z1]
broadcast.x = load1.y
broadcast.y.axis = [z0, z1]
broadcast.y.size = [s0*s1, s2]
broadcast.y.strides = [s2, 1]
sub = ascir.ops.Sub('graph1_hint/sub', graph1_hint)
sub.attr.sched.axis = [z0, z1]
sub.x1 = cast.y
sub.x2 = broadcast.y
sub.y.axis = [z0, z1]
sub.y.size = [s0*s1, s2]
sub.y.strides = [s2, 1]
exp = ascir.ops.Exp('graph1_hint/exp', graph1_hint)
exp.attr.sched.axis = [z0, z1]
exp.x = sub.y
exp.y.axis = [z0, z1]
exp.y.size = [s0*s1, s2]
exp.y.strides = [s2, 1]
store = ascir.ops.Store('graph1_hint/store', graph1_hint)
store.attr.sched.axis = [z0, z1]
store.x = exp.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, s2]
store.y.strides = [s2, 1]
output = ascir.ops.Output('graph1_hint/output', graph1_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph1_hint.infer_dtypes()
# --------------------graph2--------------------
graph2_hint = ascir.HintGraph('graph2_hint')
s0 = graph2_hint.create_size("s0")
s1 = graph2_hint.create_size("s1")
s2 = graph2_hint.create_size("s2")
z0 = graph2_hint.create_axis("z0", s0*s1)
z1 = graph2_hint.create_axis("z1", s2)
data = ascir.ops.Data('graph2_hint/data', graph2_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('graph2_hint/load', graph2_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
sum = ascir.ops.Sum('graph2_hint/sum', graph2_hint)
sum.attr.sched.axis = [z0, z1]
sum.x = load.y
sum.y.axis = [z0, z1]
sum.y.size = [s0*s1, 1]
sum.y.strides = [1, 0]
store = ascir.ops.Store('graph2_hint/store', graph2_hint)
store.attr.sched.axis = [z0, z1]
store.x = sum.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, 1]
store.y.strides = [1, 0]
output = ascir.ops.Output('graph2_hint/output', graph2_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph2_hint.infer_dtypes()
# --------------------graph3--------------------
graph3_hint = ascir.HintGraph('graph3_hint')
s0 = graph3_hint.create_size("s0")
s1 = graph3_hint.create_size("s1")
s2 = graph3_hint.create_size("s2")
z0 = graph3_hint.create_axis("z0", s0*s1)
z1 = graph3_hint.create_axis("z1", s2)
data = ascir.ops.Data('graph3_hint/data', graph3_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('graph3_hint/load', graph3_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
data1 = ascir.ops.Data('graph3_hint/data1', graph3_hint)
data1.attr.ir_attr.index = 1
data1.y.dtype = ascir.dtypes.float32
load1 = ascir.ops.Load('graph3_hint/load1', graph3_hint)
load1.attr.sched.axis = [z0, z1]
load1.attr.ir_attr.offset = 0
load1.x = data1.y
load1.y.axis = [z0, z1]
load1.y.size = [s0*s1, 1]
load1.y.strides = [1, 0]
broadcast = ascir.ops.Broadcast('graph3_hint/broadcast', graph3_hint)
broadcast.attr.sched.axis = [z0, z1]
broadcast.x = load1.y
broadcast.y.axis = [z0, z1]
broadcast.y.size = [s0*s1, s2]
broadcast.y.strides = [s2, 1]
truediv = ascir.ops.TrueDiv('graph3_hint/truediv', graph3_hint)
truediv.attr.sched.axis = [z0, z1]
truediv.x1 = load.y
truediv.x2 = broadcast.y
truediv.y.axis = [z0, z1]
truediv.y.size = [s0*s1, s2]
truediv.y.strides = [s2, 1]
cast = ascir.ops.Cast('graph3_hint/cast', graph3_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = truediv.y
cast.y.dtype = ascir.dtypes.float16
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
store = ascir.ops.Store('graph3_hint/store', graph3_hint)
store.attr.sched.axis = [z0, z1]
store.x = cast.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, s2]
store.y.strides = [s2, 1]
output = ascir.ops.Output('graph3_hint/output', graph3_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float16
graph3_hint.infer_dtypes()
# --------------------fused_graph--------------------
fused_graph = ascir.FusedGraph('fused_graph')
graph0 = ascir.ops.AscBackend('graph0', graph0_hint, fused_graph)
graph1 = ascir.ops.AscBackend('graph1', graph1_hint, fused_graph)
graph2 = ascir.ops.AscBackend('graph2', graph2_hint, fused_graph)
graph3 = ascir.ops.AscBackend('graph3', graph3_hint, fused_graph)
input0 = ascir.ops.Data('input0', fused_graph)
input0.attr.ir_attr.index = 0
workspace0 = graph0.y[0]
workspace1 = graph1.y[0]
workspace2 = graph2.y[0]
output0 = graph3.y[0]
graph0.x = [input0]
graph1.x = [input0, workspace0]
graph2.x = [workspace1]
graph3.x = [workspace1, workspace2]
graph_output0 = ascir.ops.Output('output0', fused_graph)
graph_output0.attr.ir_attr.index = 0
graph_output0.x = [output0]

fuser = Autofuser(AutofuserOptions(graph_type=1))
scheduled_fused_graph = fuser.schedule(fused_graph)
tiling_def, host_impl, device_impl = fuser.codegen(scheduled_fused_graph)
""")

    def test_stable_graph(self):
        @torch.compile(dynamic=False)
        def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x1 = x.sum(dim=1, keepdim=True)
            y1 = y.sum(dim=1, keepdim=True)
            x2 = torch.mul(x1, y1)
            return x1, x2

        with KernelCapture() as kernel_capture:
            x = torch.randn(4, 8, 8)
            y = torch.randn(4, 8, 8)
            forward(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "fused_graph"),
                                f"""# --------------------graph0--------------------
graph0_hint = ascir.HintGraph('graph0_hint')
z0 = graph0_hint.create_axis("z0", 4)
z2 = graph0_hint.create_axis("z2", 8)
z1 = graph0_hint.create_axis("z1", 8)
data = ascir.ops.Data('graph0_hint/data', graph0_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('graph0_hint/load', graph0_hint)
load.attr.sched.axis = [z0, z2, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z2, z1]
load.y.size = [4, 8, 8]
load.y.strides = [64, 8, 1]
sum = ascir.ops.Sum('graph0_hint/sum', graph0_hint)
sum.attr.sched.axis = [z0, z2, z1]
sum.x = load.y
sum.y.axis = [z0, z2, z1]
sum.y.size = [4, 1, 8]
sum.y.strides = [8, 0, 1]
store = ascir.ops.Store('graph0_hint/store', graph0_hint)
store.attr.sched.axis = [z0, z2, z1]
store.x = sum.y
store.y.axis = [z0, z2, z1]
store.y.size = [4, 1, 8]
store.y.strides = [8, 0, 1]
output = ascir.ops.Output('graph0_hint/output', graph0_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph0_hint.infer_dtypes()
# --------------------graph1--------------------
graph1_hint = ascir.HintGraph('graph1_hint')
z0 = graph1_hint.create_axis("z0", 4)
z2 = graph1_hint.create_axis("z2", 8)
z1 = graph1_hint.create_axis("z1", 8)
data = ascir.ops.Data('graph1_hint/data', graph1_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('graph1_hint/load', graph1_hint)
load.attr.sched.axis = [z0, z2, z1]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0, z2, z1]
load.y.size = [4, 8, 8]
load.y.strides = [64, 8, 1]
sum = ascir.ops.Sum('graph1_hint/sum', graph1_hint)
sum.attr.sched.axis = [z0, z2, z1]
sum.x = load.y
sum.y.axis = [z0, z2, z1]
sum.y.size = [4, 1, 8]
sum.y.strides = [8, 0, 1]
store = ascir.ops.Store('graph1_hint/store', graph1_hint)
store.attr.sched.axis = [z0, z2, z1]
store.x = sum.y
store.y.axis = [z0, z2, z1]
store.y.size = [4, 1, 8]
store.y.strides = [8, 0, 1]
output = ascir.ops.Output('graph1_hint/output', graph1_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph1_hint.infer_dtypes()
# --------------------graph2--------------------
graph2_hint = ascir.HintGraph('graph2_hint')
z0 = graph2_hint.create_axis("z0", 32)
data = ascir.ops.Data('graph2_hint/data', graph2_hint)
data.attr.ir_attr.index = 0
data.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('graph2_hint/load', graph2_hint)
load.attr.sched.axis = [z0]
load.attr.ir_attr.offset = 0
load.x = data.y
load.y.axis = [z0]
load.y.size = [32]
load.y.strides = [1]
data1 = ascir.ops.Data('graph2_hint/data1', graph2_hint)
data1.attr.ir_attr.index = 1
data1.y.dtype = ascir.dtypes.float32
load1 = ascir.ops.Load('graph2_hint/load1', graph2_hint)
load1.attr.sched.axis = [z0]
load1.attr.ir_attr.offset = 0
load1.x = data1.y
load1.y.axis = [z0]
load1.y.size = [32]
load1.y.strides = [1]
mul = ascir.ops.Mul('graph2_hint/mul', graph2_hint)
mul.attr.sched.axis = [z0]
mul.x1 = load.y
mul.x2 = load1.y
mul.y.axis = [z0]
mul.y.size = [32]
mul.y.strides = [1]
store = ascir.ops.Store('graph2_hint/store', graph2_hint)
store.attr.sched.axis = [z0]
store.x = mul.y
store.y.axis = [z0]
store.y.size = [32]
store.y.strides = [1]
output = ascir.ops.Output('graph2_hint/output', graph2_hint)
output.attr.ir_attr.index = 0
output.x = store.y
output.y.dtype = ascir.dtypes.float32
graph2_hint.infer_dtypes()
# --------------------fused_graph--------------------
fused_graph = ascir.FusedGraph('fused_graph')
graph0 = ascir.ops.AscBackend('graph0', graph0_hint, fused_graph)
graph1 = ascir.ops.AscBackend('graph1', graph1_hint, fused_graph)
graph2 = ascir.ops.AscBackend('graph2', graph2_hint, fused_graph)
input0 = ascir.ops.Data('input0', fused_graph)
input0.attr.ir_attr.index = 0
input1 = ascir.ops.Data('input1', fused_graph)
input1.attr.ir_attr.index = 1
output0 = graph0.y[0]
workspace0 = graph1.y[0]
output1 = graph2.y[0]
graph0.x = [input0]
graph1.x = [input1]
graph2.x = [output0, workspace0]
graph_output0 = ascir.ops.Output('output0', fused_graph)
graph_output0.attr.ir_attr.index = 0
graph_output0.x = [output0]
graph_output1 = ascir.ops.Output('output1', fused_graph)
graph_output1.attr.ir_attr.index = 1
graph_output1.x = [output1]

fuser = Autofuser(AutofuserOptions(graph_type=1))
scheduled_fused_graph = fuser.schedule(fused_graph)
tiling_def, host_impl, device_impl = fuser.codegen(scheduled_fused_graph)
""")


if __name__ == '__main__':
    unittest.main()
