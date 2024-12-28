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
        return sorted(self.kernels, key=lambda k: int(k.kernel_name[9:]))[index]

    def graph(self, index):
        return self.kernel(index).graph

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
        self.assertEqual(len(actual), len(expect))
        for i in range(len(actual)):
            if actual[i] != expect[i]:
                self.assertEqual(actual[i].replace('TrueDiv', 'Div').replace('truediv', 'div'),
                                 expect[i].replace('TrueDiv', 'Div').replace('truediv', 'div'))

    def test_recompile_kernel_graph(self):
        @torch.compile(dynamic=True)
        def test_abs(x):
            x = torch.abs(x)
            return x

        with KernelCapture() as kernel_capture:
            test_abs(torch.ones(6, dtype=torch.float16))
            test_abs(torch.ones(2, 4, dtype=torch.float16))

        self.assertEqual(len(kernel_capture.kernels), 2)

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
                NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
                s0 = NpuKernel0Graph.create_size("s0")
                s1 = NpuKernel0Graph.create_size("s1")
                z0 = NpuKernel0Graph.create_axis("z0", s0*s1)
                arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
                arg2_1.attr.sched.exec_order = 0
                arg2_1.y.dtype = ascir.dtypes.float16
                load = ascir.ops.Load('load', NpuKernel0Graph)
                load.attr.sched.exec_order = 1
                load.attr.sched.axis = [z0]
                load.x = arg2_1.y
                load.y.axis = [z0]
                load.y.strides = [ascir.SizeExpr(1)]
                load.y.size = [s0*s1]
                abs = ascir.ops.Abs('abs', NpuKernel0Graph)
                abs.attr.sched.exec_order = 2
                abs.attr.sched.axis = [z0]
                abs.x = load.y
                abs.y.axis = [z0]
                abs.y.strides = [ascir.SizeExpr(1)]
                abs.y.size = [s0*s1]
                store = ascir.ops.Store('store', NpuKernel0Graph)
                store.attr.sched.exec_order = 3
                store.attr.sched.axis = [z0]
                store.x = abs.y
                store.y.axis = [z0]
                store.y.strides = [ascir.SizeExpr(1)]
                store.y.size = [s0*s1]
                buf0 = ascir.ops.Output('buf0', NpuKernel0Graph)
                buf0.attr.sched.exec_order = 4
                buf0.x = store.y
                buf0.y.dtype = ascir.dtypes.float16""")

    def test_softmax_graph(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        with KernelCapture() as kernel_capture:
            x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
            test_softmax(x)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"),
                                f"""NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
            s0 = NpuKernel0Graph.create_size("s0")
            s1 = NpuKernel0Graph.create_size("s1")
            s2 = NpuKernel0Graph.create_size("s2")
            buf0_z0 = NpuKernel0Graph.create_axis("buf0_z0", s0*s1)
            buf0_z1 = NpuKernel0Graph.create_axis("buf0_z1", s2)
            buf1_z0 = NpuKernel0Graph.create_axis("buf1_z0", s0*s1)
            buf1_z1 = NpuKernel0Graph.create_axis("buf1_z1", s2)
            buf2_z0 = NpuKernel0Graph.create_axis("buf2_z0", s0*s1)
            buf2_z1 = NpuKernel0Graph.create_axis("buf2_z1", s2)
            buf3_z0 = NpuKernel0Graph.create_axis("buf3_z0", s0*s1)
            buf3_z1 = NpuKernel0Graph.create_axis("buf3_z1", s2)
            arg3_1 = ascir.ops.Data('arg3_1', NpuKernel0Graph)
            arg3_1.attr.sched.exec_order = 0
            arg3_1.y.dtype = ascir.dtypes.float16
            load = ascir.ops.Load('load', NpuKernel0Graph)
            load.attr.sched.exec_order = 1
            load.attr.sched.axis = [buf0_z0, buf0_z1]
            load.x = arg3_1.y
            load.y.axis = [buf0_z0, buf0_z1]
            load.y.strides = [s2, ascir.SizeExpr(1)]
            load.y.size = [s0*s1, s2]
            cast = ascir.ops.Cast('cast', NpuKernel0Graph)
            cast.attr.sched.exec_order = 2
            cast.attr.sched.axis = [buf0_z0, buf0_z1]
            cast.x = load.y
            cast.dst_type = ascir.dtypes.float32
            cast.y.axis = [buf0_z0, buf0_z1]
            cast.y.strides = [s2, ascir.SizeExpr(1)]
            cast.y.size = [s0*s1, s2]
            max = ascir.ops.Max('max', NpuKernel0Graph)
            max.attr.sched.exec_order = 3
            max.attr.sched.axis = [buf0_z0, buf0_z1]
            max.x = cast.y
            max.attr.hint.compute_type = 'reduce'
            max.y.axis = [buf0_z0, buf0_z1]
            max.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            max.y.size = [s0*s1, ascir.SizeExpr(1)]
            store = ascir.ops.Store('store', NpuKernel0Graph)
            store.attr.sched.exec_order = 4
            store.attr.sched.axis = [buf0_z0, buf0_z1]
            store.x = max.y
            store.y.axis = [buf0_z0, buf0_z1]
            store.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            store.y.size = [s0*s1, ascir.SizeExpr(1)]
            buf0 = ascir.ops.Workspace('buf0', NpuKernel0Graph)
            buf0.attr.sched.exec_order = 5
            buf0.x = store.y
            buf0.y.dtype = ascir.dtypes.float32
            load1 = ascir.ops.Load('load1', NpuKernel0Graph)
            load1.attr.sched.exec_order = 6
            load1.attr.sched.axis = [buf1_z0, buf1_z1]
            load1.x = arg3_1.y
            load1.y.axis = [buf1_z0, buf1_z1]
            load1.y.strides = [s2, ascir.SizeExpr(1)]
            load1.y.size = [s0*s1, s2]
            cast1 = ascir.ops.Cast('cast1', NpuKernel0Graph)
            cast1.attr.sched.exec_order = 7
            cast1.attr.sched.axis = [buf1_z0, buf1_z1]
            cast1.x = load1.y
            cast1.dst_type = ascir.dtypes.float32
            cast1.y.axis = [buf1_z0, buf1_z1]
            cast1.y.strides = [s2, ascir.SizeExpr(1)]
            cast1.y.size = [s0*s1, s2]
            load2 = ascir.ops.Load('load2', NpuKernel0Graph)
            load2.attr.sched.exec_order = 8
            load2.attr.sched.axis = [buf1_z0, buf1_z1]
            load2.x = buf0.y
            load2.y.axis = [buf1_z0, buf1_z1]
            load2.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            load2.y.size = [s0*s1, ascir.SizeExpr(1)]
            broadcast = ascir.ops.Broadcast('broadcast', NpuKernel0Graph)
            broadcast.attr.sched.exec_order = 9
            broadcast.attr.sched.axis = [buf1_z0, buf1_z1]
            broadcast.x = load2.y
            broadcast.y.axis = [buf1_z0, buf1_z1]
            broadcast.y.strides = [s2, ascir.SizeExpr(1)]
            broadcast.y.size = [s0*s1, s2]
            sub = ascir.ops.Sub('sub', NpuKernel0Graph)
            sub.attr.sched.exec_order = 10
            sub.attr.sched.axis = [buf1_z0, buf1_z1]
            sub.x1 = cast1.y
            sub.x2 = broadcast.y
            sub.y.axis = [buf1_z0, buf1_z1]
            sub.y.strides = [s2, ascir.SizeExpr(1)]
            sub.y.size = [s0*s1, s2]
            exp = ascir.ops.Exp('exp', NpuKernel0Graph)
            exp.attr.sched.exec_order = 11
            exp.attr.sched.axis = [buf1_z0, buf1_z1]
            exp.x = sub.y
            exp.y.axis = [buf1_z0, buf1_z1]
            exp.y.strides = [s2, ascir.SizeExpr(1)]
            exp.y.size = [s0*s1, s2]
            store1 = ascir.ops.Store('store1', NpuKernel0Graph)
            store1.attr.sched.exec_order = 12
            store1.attr.sched.axis = [buf1_z0, buf1_z1]
            store1.x = exp.y
            store1.y.axis = [buf1_z0, buf1_z1]
            store1.y.strides = [s2, ascir.SizeExpr(1)]
            store1.y.size = [s0*s1, s2]
            buf1 = ascir.ops.Workspace('buf1', NpuKernel0Graph)
            buf1.attr.sched.exec_order = 13
            buf1.x = store1.y
            buf1.y.dtype = ascir.dtypes.float32
            load3 = ascir.ops.Load('load3', NpuKernel0Graph)
            load3.attr.sched.exec_order = 14
            load3.attr.sched.axis = [buf2_z0, buf2_z1]
            load3.x = buf1.y
            load3.y.axis = [buf2_z0, buf2_z1]
            load3.y.strides = [s2, ascir.SizeExpr(1)]
            load3.y.size = [s0*s1, s2]
            sum = ascir.ops.Sum('sum', NpuKernel0Graph)
            sum.attr.sched.exec_order = 15
            sum.attr.sched.axis = [buf2_z0, buf2_z1]
            sum.x = load3.y
            sum.attr.hint.compute_type = 'reduce'
            sum.y.axis = [buf2_z0, buf2_z1]
            sum.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            sum.y.size = [s0*s1, ascir.SizeExpr(1)]
            store2 = ascir.ops.Store('store2', NpuKernel0Graph)
            store2.attr.sched.exec_order = 16
            store2.attr.sched.axis = [buf2_z0, buf2_z1]
            store2.x = sum.y
            store2.y.axis = [buf2_z0, buf2_z1]
            store2.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            store2.y.size = [s0*s1, ascir.SizeExpr(1)]
            buf2 = ascir.ops.Workspace('buf2', NpuKernel0Graph)
            buf2.attr.sched.exec_order = 17
            buf2.x = store2.y
            buf2.y.dtype = ascir.dtypes.float32
            load4 = ascir.ops.Load('load4', NpuKernel0Graph)
            load4.attr.sched.exec_order = 18
            load4.attr.sched.axis = [buf3_z0, buf3_z1]
            load4.x = buf1.y
            load4.y.axis = [buf3_z0, buf3_z1]
            load4.y.strides = [s2, ascir.SizeExpr(1)]
            load4.y.size = [s0*s1, s2]
            load5 = ascir.ops.Load('load5', NpuKernel0Graph)
            load5.attr.sched.exec_order = 19
            load5.attr.sched.axis = [buf3_z0, buf3_z1]
            load5.x = buf2.y
            load5.y.axis = [buf3_z0, buf3_z1]
            load5.y.strides = [ascir.SizeExpr(1), ascir.SizeExpr(0)]
            load5.y.size = [s0*s1, ascir.SizeExpr(1)]
            broadcast1 = ascir.ops.Broadcast('broadcast1', NpuKernel0Graph)
            broadcast1.attr.sched.exec_order = 20
            broadcast1.attr.sched.axis = [buf3_z0, buf3_z1]
            broadcast1.x = load5.y
            broadcast1.y.axis = [buf3_z0, buf3_z1]
            broadcast1.y.strides = [s2, ascir.SizeExpr(1)]
            broadcast1.y.size = [s0*s1, s2]
            div = ascir.ops.Div('div', NpuKernel0Graph)
            div.attr.sched.exec_order = 21
            div.attr.sched.axis = [buf3_z0, buf3_z1]
            div.x1 = load4.y
            div.x2 = broadcast1.y
            div.y.axis = [buf3_z0, buf3_z1]
            div.y.strides = [s2, ascir.SizeExpr(1)]
            div.y.size = [s0*s1, s2]
            cast2 = ascir.ops.Cast('cast2', NpuKernel0Graph)
            cast2.attr.sched.exec_order = 22
            cast2.attr.sched.axis = [buf3_z0, buf3_z1]
            cast2.x = div.y
            cast2.dst_type = ascir.dtypes.float16
            cast2.y.axis = [buf3_z0, buf3_z1]
            cast2.y.strides = [s2, ascir.SizeExpr(1)]
            cast2.y.size = [s0*s1, s2]
            store3 = ascir.ops.Store('store3', NpuKernel0Graph)
            store3.attr.sched.exec_order = 23
            store3.attr.sched.axis = [buf3_z0, buf3_z1]
            store3.x = cast2.y
            store3.y.axis = [buf3_z0, buf3_z1]
            store3.y.strides = [s2, ascir.SizeExpr(1)]
            store3.y.size = [s0*s1, s2]
            buf3 = ascir.ops.Output('buf3', NpuKernel0Graph)
            buf3.attr.sched.exec_order = 24
            buf3.x = store3.y
            buf3.y.dtype = ascir.dtypes.float16""")

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

        self.assertEqual(len(kernel_capture.kernels), 1)

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
            test_embeding(x, w)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(len(kernel_capture.graph(0).unsupported_ops) > 0)
        self.assertNotEqual(kernel_capture.graph(0).unsupported_reason, None)

    def test_size_guarded(self):
        @torch.compile(dynamic=True)
        def inference(x0, x1):
            y1 = x0 + x1
            y2 = torch.conv2d(y1, x1)
            return y2

        data1 = torch.ones(2, 2, 2, 2)
        data2 = torch.ones(2, 2, 2, 2)
        with KernelCapture() as kernel_capture:
            inference(data1, data2)

        self.assertEqual(len(kernel_capture.kernels), 2)
        output0 = kernel_capture.graph(0).ops[-1]
        output1 = kernel_capture.graph(1).ops[-1]
        self.assertEqual([str(v) for v in output0.private_attrs[f'layout.size']][:2], ['2', '2'])
        self.assertEqual([str(v) for v in output1.private_attrs[f'layout.size']][:2], ['2', '2'])

    def test_view_road_transpose_broadcast(self):
        @torch.compile(dynamic=True)
        def test_view_road_transpose_broadcast(x, y):
            return x + y.transpose(0, 2)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, 4, dtype=torch.float16)
            y = torch.ones(4, 3, 1, dtype=torch.float16)
            test_view_road_transpose_broadcast(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(kernel_capture.graph(0).get_op("broadcast"))
        self.assertTrue(kernel_capture.graph(0).get_op("transpose"))

    def test_unsupported_view_road(self):
        def test_unsupported_view_road(x):
            return torch.diagonal(x).abs()

        with KernelCapture() as kernel_capture:
            x = torch.ones(4, 4, dtype=torch.float16)
            torch.compile(dynamic=True)(test_unsupported_view_road)(x)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(kernel_capture.graph(0).get_op("reinterpretview"))

    def test_multi_moda_encoder(self):
        """
        测试dropout融合算子
        """

        def bias_dropout_add(x, bias, residual, prob, training):
            out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
            return residual + out

        @torch.compile(dynamic=True)
        def bias_dropout_add_fused_train(x, bias, residual, prob):
            return bias_dropout_add(x, bias, residual, prob, True)

        x = torch.ones(2)
        bias = torch.ones(2)
        residual = torch.ones(2)
        prob = 0.3
        with KernelCapture() as kernel_capture:
            bias_dropout_add_fused_train(x, bias, residual, prob)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(len(kernel_capture.graph(0).unsupported_ops) > 0)
        self.assertNotEqual(kernel_capture.graph(0).unsupported_reason, None)

    def test_multi_moda_encoder_allow_fallback(self):
        """
        测试dropout融合算子
        """

        def bias_dropout_add(x, bias, residual, prob, training):
            out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
            return residual + out

        @torch.compile(dynamic=True)
        def bias_dropout_add_fused_train(x, bias, residual, prob):
            return bias_dropout_add(x, bias, residual, prob, True)

        x = torch.ones(2)
        bias = torch.ones(2)
        residual = torch.ones(2)
        prob = 0.3
        with KernelCapture() as kernel_capture:
            y = bias_dropout_add_fused_train(x, bias, residual, prob)

        self.assertEqual(len(kernel_capture.kernels), 1)

    @test_with_env(ASCIR_SUPPORT_CONCAT="1")
    def test_cat_lowering(self):
        @torch.compile(dynamic=True)
        def test_cat(x, y):
            return torch.cat([x, y], dim=1)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.float16)
            y = torch.ones(2, 4, dtype=torch.float16)
            test_cat(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"),
                                """NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
                                s0 = NpuKernel0Graph.create_size("s0")
                                s1 = NpuKernel0Graph.create_size("s1")
                                s2 = NpuKernel0Graph.create_size("s2")
                                z0 = NpuKernel0Graph.create_axis("z0", s0)
                                z1 = NpuKernel0Graph.create_axis("z1", s1 + s2)
                                arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
                                arg2_1.attr.sched.exec_order = 0
                                arg2_1.y.dtype = ascir.dtypes.float16
                                load = ascir.ops.Load('load', NpuKernel0Graph)
                                load.attr.sched.exec_order = 1
                                load.attr.sched.axis = [z0, z1]
                                load.x = arg2_1.y
                                load.y.axis = [z0, z1]
                                load.y.strides = [s1, ascir.SizeExpr(1)]
                                load.y.size = [s0, s1]
                                arg4_1 = ascir.ops.Data('arg4_1', NpuKernel0Graph)
                                arg4_1.attr.sched.exec_order = 2
                                arg4_1.y.dtype = ascir.dtypes.float16
                                load1 = ascir.ops.Load('load1', NpuKernel0Graph)
                                load1.attr.sched.exec_order = 3
                                load1.attr.sched.axis = [z0, z1]
                                load1.x = arg4_1.y
                                load1.y.axis = [z0, z1]
                                load1.y.strides = [s2, ascir.SizeExpr(1)]
                                load1.y.size = [s0, s2]
                                concat = ascir.ops.Concat('concat', NpuKernel0Graph)
                                concat.attr.sched.exec_order = 4
                                concat.attr.sched.axis = [z0, z1]
                                concat.x = [load.y, load1.y]
                                concat.y.axis = [z0, z1]
                                concat.y.strides = [s1 + s2, ascir.SizeExpr(1)]
                                concat.y.size = [s0, s1 + s2]
                                store = ascir.ops.Store('store', NpuKernel0Graph)
                                store.attr.sched.exec_order = 5
                                store.attr.sched.axis = [z0, z1]
                                store.x = concat.y
                                store.y.axis = [z0, z1]
                                store.y.strides = [s1 + s2, ascir.SizeExpr(1)]
                                store.y.size = [s0, s1 + s2]
                                buf0 = ascir.ops.Output('buf0', NpuKernel0Graph)
                                buf0.attr.sched.exec_order = 6
                                buf0.x = store.y
                                buf0.y.dtype = ascir.dtypes.float16
                                """)

    @test_with_env(ASCIR_SUPPORT_CONCAT="1")
    def test_cat_fused_with_pointwise(self):
        @torch.compile(dynamic=True)
        def test_cat(x, y):
            return torch.cat([x.abs(), y.exp()], dim=1)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.float16)
            y = torch.ones(2, 4, dtype=torch.float16)
            test_cat(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)

    @test_with_env(ASCIR_SUPPORT_CONCAT="1")
    def test_cat_fused_with_broadcast_transpose(self):
        def test_cat(x, y):
            return torch.cat([x.transpose(0, 1), y.transpose(2, 1)], dim=0)

        x = torch.ones(2, 3, 1, dtype=torch.float16).expand(2, 3, 5)
        y = torch.ones(4, 5, 2, dtype=torch.float16)
        with KernelCapture() as kernel_capture:
            torch.compile(dynamic=True)(test_cat)(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assertTrue(kernel_capture.graph(0).get_op("broadcast"))
        self.assertTrue(kernel_capture.graph(0).get_op("transpose"))
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"),
                                """NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
            s0 = NpuKernel0Graph.create_size("s0")
            s1 = NpuKernel0Graph.create_size("s1")
            s2 = NpuKernel0Graph.create_size("s2")
            s3 = NpuKernel0Graph.create_size("s3")
            z0 = NpuKernel0Graph.create_axis("z0", s1 + s3)
            z1 = NpuKernel0Graph.create_axis("z1", s0)
            z2 = NpuKernel0Graph.create_axis("z2", s2)
            arg3_1 = ascir.ops.Data('arg3_1', NpuKernel0Graph)
            arg3_1.attr.sched.exec_order = 0
            arg3_1.y.dtype = ascir.dtypes.float16
            load = ascir.ops.Load('load', NpuKernel0Graph)
            load.attr.sched.exec_order = 1
            load.attr.sched.axis = [z0, z1, z2]
            load.x = arg3_1.y
            load.y.axis = [z1, z0, z2]
            load.y.strides = [s1, ascir.SizeExpr(1), ascir.SizeExpr(0)]
            load.y.size = [s0, s1, ascir.SizeExpr(1)]
            broadcast = ascir.ops.Broadcast('broadcast', NpuKernel0Graph)
            broadcast.attr.sched.exec_order = 2
            broadcast.attr.sched.axis = [z0, z1, z2]
            broadcast.x = load.y
            broadcast.y.axis = [z1, z0, z2]
            broadcast.y.strides = [s1*s2, s2, ascir.SizeExpr(1)]
            broadcast.y.size = [s0, s1, s2]
            transpose = ascir.ops.Transpose('transpose', NpuKernel0Graph)
            transpose.attr.sched.exec_order = 3
            transpose.attr.sched.axis = [z0, z1, z2]
            transpose.x = broadcast.y
            transpose.y.axis = [z0, z1, z2]
            transpose.y.strides = [s0*s2, s2, ascir.SizeExpr(1)]
            transpose.y.size = [s1, s0, s2]
            arg5_1 = ascir.ops.Data('arg5_1', NpuKernel0Graph)
            arg5_1.attr.sched.exec_order = 4
            arg5_1.y.dtype = ascir.dtypes.float16
            load1 = ascir.ops.Load('load1', NpuKernel0Graph)
            load1.attr.sched.exec_order = 5
            load1.attr.sched.axis = [z0, z1, z2]
            load1.x = arg5_1.y
            load1.y.axis = [z0, z2, z1]
            load1.y.strides = [s0*s2, s0, ascir.SizeExpr(1)]
            load1.y.size = [s3, s2, s0]
            transpose1 = ascir.ops.Transpose('transpose1', NpuKernel0Graph)
            transpose1.attr.sched.exec_order = 6
            transpose1.attr.sched.axis = [z0, z1, z2]
            transpose1.x = load1.y
            transpose1.y.axis = [z0, z1, z2]
            transpose1.y.strides = [s0*s2, s2, ascir.SizeExpr(1)]
            transpose1.y.size = [s3, s0, s2]
            concat = ascir.ops.Concat('concat', NpuKernel0Graph)
            concat.attr.sched.exec_order = 7
            concat.attr.sched.axis = [z0, z1, z2]
            concat.x = [transpose.y, transpose1.y]
            concat.y.axis = [z0, z1, z2]
            concat.y.strides = [s0*s2, s2, ascir.SizeExpr(1)]
            concat.y.size = [s1 + s3, s0, s2]
            store = ascir.ops.Store('store', NpuKernel0Graph)
            store.attr.sched.exec_order = 8
            store.attr.sched.axis = [z0, z1, z2]
            store.x = concat.y
            store.y.axis = [z0, z1, z2]
            store.y.strides = [s0*s2, s2, ascir.SizeExpr(1)]
            store.y.size = [s1 + s3, s0, s2]
            buf0 = ascir.ops.Output('buf0', NpuKernel0Graph)
            buf0.attr.sched.exec_order = 9
            buf0.x = store.y
            buf0.y.dtype = ascir.dtypes.float16
            """)

    @test_with_env(NPU_INDUCTOR_FALLBACK_INT64="1")
    def test_fallback_output_int64_box(self):
        def test_fallback(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
            return torch.sum(x)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.int32)
            y = torch.ones(4, 2, dtype=torch.bool)
            z = torch.ones(4, 2, dtype=torch.int64)
            torch.compile(test_fallback)(x, y, z)

        self.assertEqual(len(kernel_capture.kernels), 0)

    @test_with_env(ASCIR_SUPPORT_CONCAT="1")
    def test_cat_lowering_with_transpose(self):
        @torch.compile(dynamic=True)
        def test_cat(x, y):
            return torch.cat([x, y], dim=1)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.float16)
            y = torch.ones(4, 2, dtype=torch.float16)
            test_cat(x, y.t())

        self.assertEqual(len(kernel_capture.kernels), 1)
        self.assert_graph_equal(kernel_capture.graph_str(0, "NpuKernel0Graph"),
                                """NpuKernel0Graph = ascir.HintGraph('NpuKernel0Graph')
                                s0 = NpuKernel0Graph.create_size("s0")
                                s1 = NpuKernel0Graph.create_size("s1")
                                s2 = NpuKernel0Graph.create_size("s2")
                                z0 = NpuKernel0Graph.create_axis("z0", s0)
                                z1 = NpuKernel0Graph.create_axis("z1", s1 + s2)
                                arg2_1 = ascir.ops.Data('arg2_1', NpuKernel0Graph)
                                arg2_1.attr.sched.exec_order = 0
                                arg2_1.y.dtype = ascir.dtypes.float16
                                load = ascir.ops.Load('load', NpuKernel0Graph)
                                load.attr.sched.exec_order = 1
                                load.attr.sched.axis = [z0, z1]
                                load.x = arg2_1.y
                                load.y.axis = [z0, z1]
                                load.y.strides = [s1, ascir.SizeExpr(1)]
                                load.y.size = [s0, s1]
                                arg4_1 = ascir.ops.Data('arg4_1', NpuKernel0Graph)
                                arg4_1.attr.sched.exec_order = 2
                                arg4_1.y.dtype = ascir.dtypes.float16
                                load1 = ascir.ops.Load('load1', NpuKernel0Graph)
                                load1.attr.sched.exec_order = 3
                                load1.attr.sched.axis = [z0, z1]
                                load1.x = arg4_1.y
                                load1.y.axis = [z1, z0]
                                load1.y.strides = [s0, ascir.SizeExpr(1)]
                                load1.y.size = [s2, s0]
                                transpose = ascir.ops.Transpose('transpose', NpuKernel0Graph)
                                transpose.attr.sched.exec_order = 4
                                transpose.attr.sched.axis = [z0, z1]
                                transpose.x = load1.y
                                transpose.y.axis = [z0, z1]
                                transpose.y.strides = [s2, ascir.SizeExpr(1)]
                                transpose.y.size = [s0, s2]
                                concat = ascir.ops.Concat('concat', NpuKernel0Graph)
                                concat.attr.sched.exec_order = 5
                                concat.attr.sched.axis = [z0, z1]
                                concat.x = [load.y, transpose.y]
                                concat.y.axis = [z0, z1]
                                concat.y.strides = [s1 + s2, ascir.SizeExpr(1)]
                                concat.y.size = [s0, s1 + s2]
                                store = ascir.ops.Store('store', NpuKernel0Graph)
                                store.attr.sched.exec_order = 6
                                store.attr.sched.axis = [z0, z1]
                                store.x = concat.y
                                store.y.axis = [z0, z1]
                                store.y.strides = [s1 + s2, ascir.SizeExpr(1)]
                                store.y.size = [s0, s1 + s2]
                                buf0 = ascir.ops.Output('buf0', NpuKernel0Graph)
                                buf0.attr.sched.exec_order = 7
                                buf0.x = store.y
                                buf0.y.dtype = ascir.dtypes.float16
                                """)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(BuildGraphTest())

    runner = unittest.TextTestRunner()
    runner.run(suite)
