import logging
import unittest
from typing import Set

from npu_extension_for_inductor.npu import NPUKernel
import npu_extension_for_inductor
from torch._inductor.virtualized import V
import torch

logging.basicConfig(level=logging.INFO)


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
                z0 = NpuKernel0Graph.create_axis("z0", ascir.SizeExpr([s0,s1]))
                arg2_1 = ascir.ops.Data('arg2_1')
                arg2_1.attr.sched.exec_order = 0
                arg2_1.y.dtype = ascir.dtypes.float16
                load = ascir.ops.Load('load')
                load.attr.sched.exec_order = 1
                load.attr.sched.axis = [z0]
                load.x = arg2_1.y
                load.y.axis = [z0]
                load.y.strides = [ascir.SizeExpr([])]
                load.y.size = [ascir.SizeExpr([s0,s1])]
                abs = ascir.ops.Abs('abs')
                abs.attr.sched.exec_order = 2
                abs.attr.sched.axis = [z0]
                abs.x = load.y
                abs.y.axis = [z0]
                abs.y.strides = [ascir.SizeExpr([])]
                abs.y.size = [ascir.SizeExpr([s0,s1])]
                store = ascir.ops.Store('store')
                store.attr.sched.exec_order = 3
                store.attr.sched.axis = [z0]
                store.x = abs.y
                store.y.axis = [z0]
                store.y.strides = [ascir.SizeExpr([])]
                store.y.size = [ascir.SizeExpr([s0,s1])]
                buf0 = ascir.ops.Output('buf0')
                buf0.attr.sched.exec_order = 4
                buf0.x = store.y
                buf0.y.dtype = ascir.dtypes.float16
                NpuKernel0Graph.set_inputs([arg2_1])
                NpuKernel0Graph.set_outputs([buf0])""")

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
            buf0_z0 = NpuKernel0Graph.create_axis("buf0_z0", ascir.SizeExpr([s0,s1]))
            buf0_z1 = NpuKernel0Graph.create_axis("buf0_z1", ascir.SizeExpr([s2]))
            buf1_z0 = NpuKernel0Graph.create_axis("buf1_z0", ascir.SizeExpr([s0,s1]))
            buf1_z1 = NpuKernel0Graph.create_axis("buf1_z1", ascir.SizeExpr([s2]))
            buf2_z0 = NpuKernel0Graph.create_axis("buf2_z0", ascir.SizeExpr([s0,s1]))
            buf2_z1 = NpuKernel0Graph.create_axis("buf2_z1", ascir.SizeExpr([s2]))
            buf3_z0 = NpuKernel0Graph.create_axis("buf3_z0", ascir.SizeExpr([s0,s1]))
            buf3_z1 = NpuKernel0Graph.create_axis("buf3_z1", ascir.SizeExpr([s2]))
            arg3_1 = ascir.ops.Data('arg3_1')
            arg3_1.attr.sched.exec_order = 0
            arg3_1.y.dtype = ascir.dtypes.float16
            load = ascir.ops.Load('load')
            load.attr.sched.exec_order = 1
            load.attr.sched.axis = [buf0_z0, buf0_z1]
            load.x = arg3_1.y
            load.y.axis = [buf0_z0, buf0_z1]
            load.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            load.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            cast = ascir.ops.Cast('cast')
            cast.attr.sched.exec_order = 2
            cast.attr.sched.axis = [buf0_z0, buf0_z1]
            cast.x = load.y
            cast.dst_type = ascir.dtypes.float32
            cast.y.axis = [buf0_z0, buf0_z1]
            cast.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            cast.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            max = ascir.ops.Max('max')
            max.attr.sched.exec_order = 3
            max.attr.sched.axis = [buf0_z0, buf0_z1]
            max.x = cast.y
            max.attr.hint.compute_type = 'reduce'
            max.y.axis = [buf0_z0, buf0_z1]
            max.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            max.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            store = ascir.ops.Store('store')
            store.attr.sched.exec_order = 4
            store.attr.sched.axis = [buf0_z0, buf0_z1]
            store.x = max.y
            store.y.axis = [buf0_z0, buf0_z1]
            store.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            store.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            buf0 = ascir.ops.Workspace('buf0')
            buf0.attr.sched.exec_order = 5
            buf0.x = store.y
            buf0.y.dtype = ascir.dtypes.float32
            load1 = ascir.ops.Load('load1')
            load1.attr.sched.exec_order = 6
            load1.attr.sched.axis = [buf1_z0, buf1_z1]
            load1.x = arg3_1.y
            load1.y.axis = [buf1_z0, buf1_z1]
            load1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            load1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            cast1 = ascir.ops.Cast('cast1')
            cast1.attr.sched.exec_order = 7
            cast1.attr.sched.axis = [buf1_z0, buf1_z1]
            cast1.x = load1.y
            cast1.dst_type = ascir.dtypes.float32
            cast1.y.axis = [buf1_z0, buf1_z1]
            cast1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            cast1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            load2 = ascir.ops.Load('load2')
            load2.attr.sched.exec_order = 8
            load2.attr.sched.axis = [buf1_z0, buf1_z1]
            load2.x = buf0.y
            load2.y.axis = [buf1_z0, buf1_z1]
            load2.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            load2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            broadcast = ascir.ops.Broadcast('broadcast')
            broadcast.attr.sched.exec_order = 9
            broadcast.attr.sched.axis = [buf1_z0, buf1_z1]
            broadcast.x = load2.y
            broadcast.y.axis = [buf1_z0, buf1_z1]
            broadcast.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            broadcast.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            sub = ascir.ops.Sub('sub')
            sub.attr.sched.exec_order = 10
            sub.attr.sched.axis = [buf1_z0, buf1_z1]
            sub.x1 = cast1.y
            sub.x2 = broadcast.y
            sub.y.axis = [buf1_z0, buf1_z1]
            sub.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            sub.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            exp = ascir.ops.Exp('exp')
            exp.attr.sched.exec_order = 11
            exp.attr.sched.axis = [buf1_z0, buf1_z1]
            exp.x = sub.y
            exp.y.axis = [buf1_z0, buf1_z1]
            exp.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            exp.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            store1 = ascir.ops.Store('store1')
            store1.attr.sched.exec_order = 12
            store1.attr.sched.axis = [buf1_z0, buf1_z1]
            store1.x = exp.y
            store1.y.axis = [buf1_z0, buf1_z1]
            store1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            store1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            buf1 = ascir.ops.Workspace('buf1')
            buf1.attr.sched.exec_order = 13
            buf1.x = store1.y
            buf1.y.dtype = ascir.dtypes.float32
            load3 = ascir.ops.Load('load3')
            load3.attr.sched.exec_order = 14
            load3.attr.sched.axis = [buf2_z0, buf2_z1]
            load3.x = buf1.y
            load3.y.axis = [buf2_z0, buf2_z1]
            load3.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            load3.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            sum = ascir.ops.Sum('sum')
            sum.attr.sched.exec_order = 15
            sum.attr.sched.axis = [buf2_z0, buf2_z1]
            sum.x = load3.y
            sum.attr.hint.compute_type = 'reduce'
            sum.y.axis = [buf2_z0, buf2_z1]
            sum.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            sum.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            store2 = ascir.ops.Store('store2')
            store2.attr.sched.exec_order = 16
            store2.attr.sched.axis = [buf2_z0, buf2_z1]
            store2.x = sum.y
            store2.y.axis = [buf2_z0, buf2_z1]
            store2.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            store2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            buf2 = ascir.ops.Workspace('buf2')
            buf2.attr.sched.exec_order = 17
            buf2.x = store2.y
            buf2.y.dtype = ascir.dtypes.float32
            load4 = ascir.ops.Load('load4')
            load4.attr.sched.exec_order = 18
            load4.attr.sched.axis = [buf3_z0, buf3_z1]
            load4.x = buf1.y
            load4.y.axis = [buf3_z0, buf3_z1]
            load4.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            load4.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            load5 = ascir.ops.Load('load5')
            load5.attr.sched.exec_order = 19
            load5.attr.sched.axis = [buf3_z0, buf3_z1]
            load5.x = buf2.y
            load5.y.axis = [buf3_z0, buf3_z1]
            load5.y.strides = [ascir.SizeExpr([]), ascir.SizeExpr([0])]
            load5.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([])]
            broadcast1 = ascir.ops.Broadcast('broadcast1')
            broadcast1.attr.sched.exec_order = 20
            broadcast1.attr.sched.axis = [buf3_z0, buf3_z1]
            broadcast1.x = load5.y
            broadcast1.y.axis = [buf3_z0, buf3_z1]
            broadcast1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            broadcast1.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            div = ascir.ops.Div('div')
            div.attr.sched.exec_order = 21
            div.attr.sched.axis = [buf3_z0, buf3_z1]
            div.x1 = load4.y
            div.x2 = broadcast1.y
            div.y.axis = [buf3_z0, buf3_z1]
            div.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            div.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            cast2 = ascir.ops.Cast('cast2')
            cast2.attr.sched.exec_order = 22
            cast2.attr.sched.axis = [buf3_z0, buf3_z1]
            cast2.x = div.y
            cast2.dst_type = ascir.dtypes.float16
            cast2.y.axis = [buf3_z0, buf3_z1]
            cast2.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            cast2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            store3 = ascir.ops.Store('store3')
            store3.attr.sched.exec_order = 23
            store3.attr.sched.axis = [buf3_z0, buf3_z1]
            store3.x = cast2.y
            store3.y.axis = [buf3_z0, buf3_z1]
            store3.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
            store3.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
            buf3 = ascir.ops.Output('buf3')
            buf3.attr.sched.exec_order = 24
            buf3.x = store3.y
            buf3.y.dtype = ascir.dtypes.float16
            NpuKernel0Graph.set_inputs([arg3_1])
            NpuKernel0Graph.set_outputs([buf3])""")

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
                                z0 = NpuKernel0Graph.create_axis("z0", ascir.SizeExpr([s0]))
                                z1 = NpuKernel0Graph.create_axis("z1", ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]))
                                arg2_1 = ascir.ops.Data('arg2_1')
                                arg2_1.attr.sched.exec_order = 0
                                arg2_1.y.dtype = ascir.dtypes.float16
                                load = ascir.ops.Load('load')
                                load.attr.sched.exec_order = 1
                                load.attr.sched.axis = [z0, z1]
                                load.x = arg2_1.y
                                load.y.axis = [z0, z1]
                                load.y.strides = [ascir.SizeExpr([s1]), ascir.SizeExpr([])]
                                load.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1])]
                                arg4_1 = ascir.ops.Data('arg4_1')
                                arg4_1.attr.sched.exec_order = 2
                                arg4_1.y.dtype = ascir.dtypes.float16
                                load1 = ascir.ops.Load('load1')
                                load1.attr.sched.exec_order = 3
                                load1.attr.sched.axis = [z0, z1]
                                load1.x = arg4_1.y
                                load1.y.axis = [z0, z1]
                                load1.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                load1.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s2])]
                                concat = ascir.ops.Concat('concat')
                                concat.attr.sched.exec_order = 4
                                concat.attr.sched.axis = [z0, z1]
                                concat.x = [load.y, load1.y]
                                concat.y.axis = [z0, z1]
                                concat.y.strides = [ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                concat.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]) + ascir.SizeExpr([s2])]
                                store = ascir.ops.Store('store')
                                store.attr.sched.exec_order = 5
                                store.attr.sched.axis = [z0, z1]
                                store.x = concat.y
                                store.y.axis = [z0, z1]
                                store.y.strides = [ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                store.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]) + ascir.SizeExpr([s2])]
                                buf0 = ascir.ops.Output('buf0')
                                buf0.attr.sched.exec_order = 6
                                buf0.x = store.y
                                buf0.y.dtype = ascir.dtypes.float16
                                NpuKernel0Graph.set_inputs([arg2_1, arg4_1])
                                NpuKernel0Graph.set_outputs([buf0])
                                """)

    def test_cat_fused_with_pointwise(self):
        @torch.compile(dynamic=True)
        def test_cat(x, y):
            return torch.cat([x.abs(), y.exp()], dim=1)

        with KernelCapture() as kernel_capture:
            x = torch.ones(2, 3, dtype=torch.float16)
            y = torch.ones(2, 4, dtype=torch.float16)
            test_cat(x, y)

        self.assertEqual(len(kernel_capture.kernels), 1)

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
                                z0 = NpuKernel0Graph.create_axis("z0", ascir.SizeExpr([s0]))
                                z1 = NpuKernel0Graph.create_axis("z1", ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]))
                                arg2_1 = ascir.ops.Data('arg2_1')
                                arg2_1.attr.sched.exec_order = 0
                                arg2_1.y.dtype = ascir.dtypes.float16
                                load = ascir.ops.Load('load')
                                load.attr.sched.exec_order = 1
                                load.attr.sched.axis = [z0, z1]
                                load.x = arg2_1.y
                                load.y.axis = [z0, z1]
                                load.y.strides = [ascir.SizeExpr([s1]), ascir.SizeExpr([])]
                                load.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1])]
                                arg4_1 = ascir.ops.Data('arg4_1')
                                arg4_1.attr.sched.exec_order = 2
                                arg4_1.y.dtype = ascir.dtypes.float16
                                load1 = ascir.ops.Load('load1')
                                load1.attr.sched.exec_order = 3
                                load1.attr.sched.axis = [z0, z1]
                                load1.x = arg4_1.y
                                load1.y.axis = [z1, z0]
                                load1.y.strides = [ascir.SizeExpr([s0]), ascir.SizeExpr([])]
                                load1.y.size = [ascir.SizeExpr([s2]), ascir.SizeExpr([s0])]
                                transpose = ascir.ops.Transpose('transpose')
                                transpose.attr.sched.exec_order = 4
                                transpose.attr.sched.axis = [z0, z1]
                                transpose.x0 = load1.y
                                transpose.y.axis = [z0, z1]
                                transpose.y.strides = [ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                transpose.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s2])]
                                concat = ascir.ops.Concat('concat')
                                concat.attr.sched.exec_order = 5
                                concat.attr.sched.axis = [z0, z1]
                                concat.x = [load.y, transpose.y]
                                concat.y.axis = [z0, z1]
                                concat.y.strides = [ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                concat.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]) + ascir.SizeExpr([s2])]
                                store = ascir.ops.Store('store')
                                store.attr.sched.exec_order = 6
                                store.attr.sched.axis = [z0, z1]
                                store.x = concat.y
                                store.y.axis = [z0, z1]
                                store.y.strides = [ascir.SizeExpr([s1]) + ascir.SizeExpr([s2]), ascir.SizeExpr([])]
                                store.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]) + ascir.SizeExpr([s2])]
                                buf0 = ascir.ops.Output('buf0')
                                buf0.attr.sched.exec_order = 7
                                buf0.x = store.y
                                buf0.y.dtype = ascir.dtypes.float16
                                NpuKernel0Graph.set_inputs([arg2_1, arg4_1])
                                NpuKernel0Graph.set_outputs([buf0])
                                """)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(BuildGraphTest())

    runner = unittest.TextTestRunner()
    runner.run(suite)
