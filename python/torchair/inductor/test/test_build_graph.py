import contextlib
import os
import unittest
from typing import List, Set

import npu_extension_for_inductor
from npu_extension_for_inductor.npu import NPUKernel
import torch
from torch._inductor.virtualized import V


@contextlib.contextmanager
def disable_npu_fallback(disable=True):
    old = os.getenv("DISABLE_NPU_FALLBACK", "0")
    try:
        os.environ["DISABLE_NPU_FALLBACK"] = "1" if disable else "0"
        yield
    finally:
        os.environ["DISABLE_NPU_FALLBACK"] = old


class KernelCapture:
    def __init__(self):
        self.kernels: Set[NPUKernel] = set()
        self.origin = V.set_kernel_handler

    def watch_npu_kernel(self, kernel):
        if type(kernel) == NPUKernel:
            self.kernels.add(kernel)
        return self.origin(kernel)

    def kernel(self, index):
        return list(self.kernels)[index]

    def graph(self, index):
        return list(self.kernels)[index].graph

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
        print(f"Capture message:\n{exc_val}", flush=True)
        return True


class BuildGraphTest(unittest.TestCase):
    def assert_graph_equal(self, actual, expect):
        actual = [line.strip() for line in actual.split('\n') if line.strip()]
        expect = [line.strip() for line in expect.split('\n') if line.strip()]
        self.assertEqual(len(actual), len(expect))
        for i in range(len(actual)):
            if actual[i] != expect[i]:
                self.assertEqual(actual[i].replace('TrueDiv', 'Div').replace('truediv', 'div'),
                                 expect[i].replace('TrueDiv', 'Div').replace('truediv', 'div'))

    def test_abs_graph(self):
        @torch.compile(dynamic=True)
        def test_abs(x):
            x = torch.abs(x)
            return x

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.float16)
            test_abs(x)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"), f"""
            def NpuKernel0Graph():

                import os
                if os.getenv('ASCIR_NOT_READY', None) == "1":
                    return None
                from pyautofuse import ascir
                NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
                s0 = NpuKernel0Graph.create_size("s0")
                s1 = NpuKernel0Graph.create_size("s1")
                z0 = NpuKernel0Graph.create_axis("z0", ascir.SizeExpr([s0,s1]))
                size_vars = ascir.ops.Data('size_vars')
                size_vars.attr.sched.exec_order = 0
                size_vars.attr.sched.axis = []
                size_vars.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1])]
                arg2_1 = ascir.ops.Data('arg2_1')
                arg2_1.attr.sched.exec_order = 1
                arg2_1.attr.sched.axis = [z0]
                arg2_1.y.size = [ascir.SizeExpr([s0,s1])]
                arg2_1.y.dtype = ascir.dtypes.float16
                arg2_1.y.axis = [z0]
                arg2_1.y.strides = [ascir.SizeExpr([])]
                load = ascir.ops.Load('load')
                load.attr.sched.exec_order = 2
                load.attr.sched.axis = [z0]
                load.x = arg2_1.y
                load.y.axis = [z0]
                load.y.strides = [ascir.SizeExpr([])]
                load.y.size = [ascir.SizeExpr([s0,s1])]
                abs = ascir.ops.Abs('abs')
                abs.attr.sched.exec_order = 3
                abs.attr.sched.axis = [z0]
                abs.x = load.y
                abs.y.axis = [z0]
                abs.y.strides = [ascir.SizeExpr([])]
                abs.y.size = [ascir.SizeExpr([s0,s1])]
                store = ascir.ops.Store('store')
                store.attr.sched.exec_order = 4
                store.attr.sched.axis = [z0]
                store.x = abs.y
                store.y.axis = [z0]
                store.y.strides = [ascir.SizeExpr([])]
                store.y.size = [ascir.SizeExpr([s0,s1])]
                buf0 = ascir.ops.Output('buf0')
                buf0.attr.sched.exec_order = 5
                buf0.attr.sched.axis = [z0]
                buf0.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1])]
                buf0.y.dtype = ascir.dtypes.float16
                buf0.x = store.y
            
                NpuKernel0Graph.set_inputs([size_vars, arg2_1])
                NpuKernel0Graph.set_outputs([buf0])
                return NpuKernel0Graph
            """)

    def test_softmax_graph(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        with KernelCapture() as kernel_capture:
            x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
            test_softmax(x)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"), f"""
            def NpuKernel0Graph():

                import os
                if os.getenv('ASCIR_NOT_READY', None) == "1":
                    return None
                from pyautofuse import ascir
                NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
                s0 = NpuKernel0Graph.create_size("s0")
                s1 = NpuKernel0Graph.create_size("s1")
                s2 = NpuKernel0Graph.create_size("s2")
                z0 = NpuKernel0Graph.create_axis("z0", ascir.SizeExpr([s0,s1]))
                z1 = NpuKernel0Graph.create_axis("z1", ascir.SizeExpr([s2]))
                size_vars = ascir.ops.Data('size_vars')
                size_vars.attr.sched.exec_order = 0
                size_vars.attr.sched.axis = []
                size_vars.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
                arg3_1 = ascir.ops.Data('arg3_1')
                arg3_1.attr.sched.exec_order = 1
                arg3_1.attr.sched.axis = [z0, z1]
                arg3_1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                arg3_1.y.dtype = ascir.dtypes.float16
                arg3_1.y.axis = [z0, z1]
                arg3_1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load = ascir.ops.Load('load')
                load.attr.sched.exec_order = 2
                load.attr.sched.axis = [z0, z1]
                load.x = arg3_1.y
                load.y.axis = [z0, z1]
                load.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                cast = ascir.ops.Cast('cast')
                cast.attr.sched.exec_order = 3
                cast.attr.sched.axis = [z0, z1]
                cast.x = load.y
                cast.dst_type = ascir.dtypes.float32
                cast.y.axis = [z0, z1]
                cast.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                cast.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                max = ascir.ops.Max('max')
                max.attr.sched.exec_order = 4
                max.attr.sched.axis = [z0, z1]
                max.x = cast.y
                max.attr.hint.compute_type = 'reduce'
                max.y.axis = [z0, z1]
                max.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                max.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                store = ascir.ops.Store('store')
                store.attr.sched.exec_order = 5
                store.attr.sched.axis = [z0, z1]
                store.x = max.y
                store.y.axis = [z0, z1]
                store.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                store.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                buf0 = ascir.ops.Workspace('buf0')
                buf0.attr.sched.exec_order = 6
                buf0.attr.sched.axis = [z0, z1]
                buf0.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                buf0.y.dtype = ascir.dtypes.float32
                buf0.x = store.y
                buf0.y.axis = [z0, z1]
                buf0.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                load1 = ascir.ops.Load('load1')
                load1.attr.sched.exec_order = 7
                load1.attr.sched.axis = [z0, z1]
                load1.x = arg3_1.y
                load1.y.axis = [z0, z1]
                load1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                cast1 = ascir.ops.Cast('cast1')
                cast1.attr.sched.exec_order = 8
                cast1.attr.sched.axis = [z0, z1]
                cast1.x = load1.y
                cast1.dst_type = ascir.dtypes.float32
                cast1.y.axis = [z0, z1]
                cast1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                cast1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                load2 = ascir.ops.Load('load2')
                load2.attr.sched.exec_order = 9
                load2.attr.sched.axis = [z0, z1]
                load2.x = buf0.y
                load2.y.axis = [z0, z1]
                load2.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                load2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                broadcast = ascir.ops.Broadcast('broadcast')
                broadcast.attr.sched.exec_order = 10
                broadcast.attr.sched.axis = [z0, z1]
                broadcast.x = load2.y
                broadcast.y.axis = [z0, z1]
                broadcast.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                broadcast.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                sub = ascir.ops.Sub('sub')
                sub.attr.sched.exec_order = 11
                sub.attr.sched.axis = [z0, z1]
                sub.x1 = cast1.y
                sub.x2 = broadcast.y
                sub.y.axis = [z0, z1]
                sub.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                sub.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                exp = ascir.ops.Exp('exp')
                exp.attr.sched.exec_order = 12
                exp.attr.sched.axis = [z0, z1]
                exp.x = sub.y
                exp.y.axis = [z0, z1]
                exp.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                exp.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                store1 = ascir.ops.Store('store1')
                store1.attr.sched.exec_order = 13
                store1.attr.sched.axis = [z0, z1]
                store1.x = exp.y
                store1.y.axis = [z0, z1]
                store1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                store1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                buf1 = ascir.ops.Workspace('buf1')
                buf1.attr.sched.exec_order = 14
                buf1.attr.sched.axis = [z0, z1]
                buf1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                buf1.y.dtype = ascir.dtypes.float32
                buf1.x = store1.y
                buf1.y.axis = [z0, z1]
                buf1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load3 = ascir.ops.Load('load3')
                load3.attr.sched.exec_order = 15
                load3.attr.sched.axis = [z0, z1]
                load3.x = buf1.y
                load3.y.axis = [z0, z1]
                load3.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load3.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                sum = ascir.ops.Sum('sum')
                sum.attr.sched.exec_order = 16
                sum.attr.sched.axis = [z0, z1]
                sum.x = load3.y
                sum.attr.hint.compute_type = 'reduce'
                sum.y.axis = [z0, z1]
                sum.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                sum.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                store2 = ascir.ops.Store('store2')
                store2.attr.sched.exec_order = 17
                store2.attr.sched.axis = [z0, z1]
                store2.x = sum.y
                store2.y.axis = [z0, z1]
                store2.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                store2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                buf2 = ascir.ops.Workspace('buf2')
                buf2.attr.sched.exec_order = 18
                buf2.attr.sched.axis = [z0, z1]
                buf2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                buf2.y.dtype = ascir.dtypes.float32
                buf2.x = store2.y
                buf2.y.axis = [z0, z1]
                buf2.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                load4 = ascir.ops.Load('load4')
                load4.attr.sched.exec_order = 19
                load4.attr.sched.axis = [z0, z1]
                load4.x = buf1.y
                load4.y.axis = [z0, z1]
                load4.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                load4.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                load5 = ascir.ops.Load('load5')
                load5.attr.sched.exec_order = 20
                load5.attr.sched.axis = [z0, z1]
                load5.x = buf2.y
                load5.y.axis = [z0, z1]
                load5.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
                load5.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
                broadcast1 = ascir.ops.Broadcast('broadcast1')
                broadcast1.attr.sched.exec_order = 21
                broadcast1.attr.sched.axis = [z0, z1]
                broadcast1.x = load5.y
                broadcast1.y.axis = [z0, z1]
                broadcast1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                broadcast1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                truediv = ascir.ops.TrueDiv('truediv')
                truediv.attr.sched.exec_order = 22
                truediv.attr.sched.axis = [z0, z1]
                truediv.x1 = load4.y
                truediv.x2 = broadcast1.y
                truediv.y.axis = [z0, z1]
                truediv.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                truediv.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                cast2 = ascir.ops.Cast('cast2')
                cast2.attr.sched.exec_order = 23
                cast2.attr.sched.axis = [z0, z1]
                cast2.x = truediv.y
                cast2.dst_type = ascir.dtypes.float16
                cast2.y.axis = [z0, z1]
                cast2.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                cast2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                store3 = ascir.ops.Store('store3')
                store3.attr.sched.exec_order = 24
                store3.attr.sched.axis = [z0, z1]
                store3.x = cast2.y
                store3.y.axis = [z0, z1]
                store3.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                store3.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
                buf3 = ascir.ops.Output('buf3')
                buf3.attr.sched.exec_order = 25
                buf3.attr.sched.axis = [z0, z1]
                buf3.y.size = [ascir.SizeExpr([]), ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
                buf3.y.dtype = ascir.dtypes.float16
                buf3.x = store3.y
            
                NpuKernel0Graph.set_inputs([size_vars, arg3_1])
                NpuKernel0Graph.set_outputs([buf3])
                return NpuKernel0Graph""")

    def test_embeding_fallback(self):
        """
        测试带有embeding算子的图
        """

        @torch.compile(dynamic=True)
        def test_embeding(x, w):
            x = torch.nn.functional.embedding(x, w)
            return x

        x = torch.ones(2, dtype=torch.int64)
        w = torch.arange(0, 200, dtype=torch.float16).view(10, 20)
        with KernelCapture() as kernel_capture:
            y = test_embeding(x, w)

        self.assertEqual(len(kernel_capture.kernels), 0)

    def test_disable_fallback(self):
        """
        测试带有embeding算子的图
        """

        @torch.compile(dynamic=True)
        def test_embeding(x, w):
            x = torch.nn.functional.embedding(x, w)
            return x

        x = torch.ones(2, dtype=torch.int64)
        w = torch.arange(0, 200, dtype=torch.float16).view(10, 20)
        with KernelCapture() as kernel_capture:
            with disable_npu_fallback():
                y = test_embeding(x, w)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(len(kernel_capture.graph(0).unsupported_ops) > 0)
        self.assertNotEqual(kernel_capture.graph(0).unsupported_reason, None)


if __name__ == '__main__':
    unittest.main()
