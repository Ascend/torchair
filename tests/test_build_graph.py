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
        return sorted(self.kernels, key=lambda k: int(k.kernel_name[10:]))[index]

    def graph(self, index):
        return self.kernel(index).fused_graph

    def graph_str(self, index, replace_name):
        graph = self.graph(index)
        return graph.codegen(graph.name).getvalue().replace(graph.name, replace_name)

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
                                f"""# --------------------buf0_asc--------------------
buf0_asc_hint = ascir.HintGraph('buf0_asc_hint')
s0 = buf0_asc_hint.create_size("s0")
s1 = buf0_asc_hint.create_size("s1")
s2 = buf0_asc_hint.create_size("s2")
z0 = buf0_asc_hint.create_axis("z0", s0*s1)
z1 = buf0_asc_hint.create_axis("z1", s2)
arg3_1 = ascir.ops.Data('buf0_asc_hint/arg3_1', buf0_asc_hint)
arg3_1.attr.ir_attr.index = 0
arg3_1.y.dtype = ascir.dtypes.float16
load = ascir.ops.Load('buf0_asc_hint/load', buf0_asc_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = arg3_1.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
cast = ascir.ops.Cast('buf0_asc_hint/cast', buf0_asc_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = load.y
cast.dst_type = ascir.dtypes.float32
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
max = ascir.ops.Max('buf0_asc_hint/max', buf0_asc_hint)
max.attr.sched.axis = [z0, z1]
max.x = cast.y
max.attr.hint.compute_type = 'reduce'
max.y.axis = [z0, z1]
max.y.size = [s0*s1, 1]
max.y.strides = [1, 0]
store = ascir.ops.Store('buf0_asc_hint/store', buf0_asc_hint)
store.attr.sched.axis = [z0, z1]
store.x = max.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, 1]
store.y.strides = [1, 0]
buf0 = ascir.ops.Output('buf0_asc_hint/buf0', buf0_asc_hint)
buf0.attr.ir_attr.index = 0
buf0.x = store.y
buf0.y.dtype = ascir.dtypes.float32
# --------------------buf1_asc--------------------
buf1_asc_hint = ascir.HintGraph('buf1_asc_hint')
s0 = buf1_asc_hint.create_size("s0")
s1 = buf1_asc_hint.create_size("s1")
s2 = buf1_asc_hint.create_size("s2")
z0 = buf1_asc_hint.create_axis("z0", s0*s1)
z1 = buf1_asc_hint.create_axis("z1", s2)
arg3_1 = ascir.ops.Data('buf1_asc_hint/arg3_1', buf1_asc_hint)
arg3_1.attr.ir_attr.index = 0
arg3_1.y.dtype = ascir.dtypes.float16
load = ascir.ops.Load('buf1_asc_hint/load', buf1_asc_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = arg3_1.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
cast = ascir.ops.Cast('buf1_asc_hint/cast', buf1_asc_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = load.y
cast.dst_type = ascir.dtypes.float32
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
buf0 = ascir.ops.Data('buf1_asc_hint/buf0', buf1_asc_hint)
buf0.attr.ir_attr.index = 1
buf0.y.dtype = ascir.dtypes.float32
load1 = ascir.ops.Load('buf1_asc_hint/load1', buf1_asc_hint)
load1.attr.sched.axis = [z0, z1]
load1.attr.ir_attr.offset = 0
load1.x = buf0.y
load1.y.axis = [z0, z1]
load1.y.size = [s0*s1, 1]
load1.y.strides = [1, 0]
broadcast = ascir.ops.Broadcast('buf1_asc_hint/broadcast', buf1_asc_hint)
broadcast.attr.sched.axis = [z0, z1]
broadcast.x = load1.y
broadcast.y.axis = [z0, z1]
broadcast.y.size = [s0*s1, s2]
broadcast.y.strides = [s2, 1]
sub = ascir.ops.Sub('buf1_asc_hint/sub', buf1_asc_hint)
sub.attr.sched.axis = [z0, z1]
sub.x1 = cast.y
sub.x2 = broadcast.y
sub.y.axis = [z0, z1]
sub.y.size = [s0*s1, s2]
sub.y.strides = [s2, 1]
exp = ascir.ops.Exp('buf1_asc_hint/exp', buf1_asc_hint)
exp.attr.sched.axis = [z0, z1]
exp.x = sub.y
exp.y.axis = [z0, z1]
exp.y.size = [s0*s1, s2]
exp.y.strides = [s2, 1]
store = ascir.ops.Store('buf1_asc_hint/store', buf1_asc_hint)
store.attr.sched.axis = [z0, z1]
store.x = exp.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, s2]
store.y.strides = [s2, 1]
buf1 = ascir.ops.Output('buf1_asc_hint/buf1', buf1_asc_hint)
buf1.attr.ir_attr.index = 0
buf1.x = store.y
buf1.y.dtype = ascir.dtypes.float32
# --------------------buf2_asc--------------------
buf2_asc_hint = ascir.HintGraph('buf2_asc_hint')
s0 = buf2_asc_hint.create_size("s0")
s1 = buf2_asc_hint.create_size("s1")
s2 = buf2_asc_hint.create_size("s2")
z0 = buf2_asc_hint.create_axis("z0", s0*s1)
z1 = buf2_asc_hint.create_axis("z1", s2)
buf1 = ascir.ops.Data('buf2_asc_hint/buf1', buf2_asc_hint)
buf1.attr.ir_attr.index = 0
buf1.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('buf2_asc_hint/load', buf2_asc_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = buf1.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
sum = ascir.ops.Sum('buf2_asc_hint/sum', buf2_asc_hint)
sum.attr.sched.axis = [z0, z1]
sum.x = load.y
sum.attr.hint.compute_type = 'reduce'
sum.y.axis = [z0, z1]
sum.y.size = [s0*s1, 1]
sum.y.strides = [1, 0]
store = ascir.ops.Store('buf2_asc_hint/store', buf2_asc_hint)
store.attr.sched.axis = [z0, z1]
store.x = sum.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, 1]
store.y.strides = [1, 0]
buf2 = ascir.ops.Output('buf2_asc_hint/buf2', buf2_asc_hint)
buf2.attr.ir_attr.index = 0
buf2.x = store.y
buf2.y.dtype = ascir.dtypes.float32
# --------------------buf3_asc--------------------
buf3_asc_hint = ascir.HintGraph('buf3_asc_hint')
s0 = buf3_asc_hint.create_size("s0")
s1 = buf3_asc_hint.create_size("s1")
s2 = buf3_asc_hint.create_size("s2")
z0 = buf3_asc_hint.create_axis("z0", s0*s1)
z1 = buf3_asc_hint.create_axis("z1", s2)
buf1 = ascir.ops.Data('buf3_asc_hint/buf1', buf3_asc_hint)
buf1.attr.ir_attr.index = 0
buf1.y.dtype = ascir.dtypes.float32
load = ascir.ops.Load('buf3_asc_hint/load', buf3_asc_hint)
load.attr.sched.axis = [z0, z1]
load.attr.ir_attr.offset = 0
load.x = buf1.y
load.y.axis = [z0, z1]
load.y.size = [s0*s1, s2]
load.y.strides = [s2, 1]
buf2 = ascir.ops.Data('buf3_asc_hint/buf2', buf3_asc_hint)
buf2.attr.ir_attr.index = 1
buf2.y.dtype = ascir.dtypes.float32
load1 = ascir.ops.Load('buf3_asc_hint/load1', buf3_asc_hint)
load1.attr.sched.axis = [z0, z1]
load1.attr.ir_attr.offset = 0
load1.x = buf2.y
load1.y.axis = [z0, z1]
load1.y.size = [s0*s1, 1]
load1.y.strides = [1, 0]
broadcast = ascir.ops.Broadcast('buf3_asc_hint/broadcast', buf3_asc_hint)
broadcast.attr.sched.axis = [z0, z1]
broadcast.x = load1.y
broadcast.y.axis = [z0, z1]
broadcast.y.size = [s0*s1, s2]
broadcast.y.strides = [s2, 1]
truediv = ascir.ops.TrueDiv('buf3_asc_hint/truediv', buf3_asc_hint)
truediv.attr.sched.axis = [z0, z1]
truediv.x1 = load.y
truediv.x2 = broadcast.y
truediv.y.axis = [z0, z1]
truediv.y.size = [s0*s1, s2]
truediv.y.strides = [s2, 1]
cast = ascir.ops.Cast('buf3_asc_hint/cast', buf3_asc_hint)
cast.attr.sched.axis = [z0, z1]
cast.x = truediv.y
cast.dst_type = ascir.dtypes.float16
cast.y.axis = [z0, z1]
cast.y.size = [s0*s1, s2]
cast.y.strides = [s2, 1]
store = ascir.ops.Store('buf3_asc_hint/store', buf3_asc_hint)
store.attr.sched.axis = [z0, z1]
store.x = cast.y
store.y.axis = [z0, z1]
store.y.size = [s0*s1, s2]
store.y.strides = [s2, 1]
buf3 = ascir.ops.Output('buf3_asc_hint/buf3', buf3_asc_hint)
buf3.attr.ir_attr.index = 0
buf3.x = store.y
buf3.y.dtype = ascir.dtypes.float16
# --------------------fused_graph--------------------
fused_graph = ascir.FusedGraph('fused_graph')
buf0_asc = ascir.ops.AscGraph('buf0_asc', buf0_asc_hint, fused_graph)
buf1_asc = ascir.ops.AscGraph('buf1_asc', buf1_asc_hint, fused_graph)
buf2_asc = ascir.ops.AscGraph('buf2_asc', buf2_asc_hint, fused_graph)
buf3_asc = ascir.ops.AscGraph('buf3_asc', buf3_asc_hint, fused_graph)
arg3_1 = ascir.ops.Data('arg3_1', fused_graph)
arg3_1.attr.ir_attr.index = 0
buf0 = buf0_asc.y[0]
buf1 = buf1_asc.y[0]
buf2 = buf2_asc.y[0]
buf3 = buf3_asc.y[0]
buf0_asc.x = [arg3_1]
buf1_asc.x = [arg3_1, buf0]
buf2_asc.x = [buf1]
buf3_asc.x = [buf1, buf2]
buf3_output = ascir.ops.Output('buf3', fused_graph)
buf3_output.attr.ir_attr.index = 0
buf3_output.x = [buf3]""")


if __name__ == '__main__':
    unittest.main()
