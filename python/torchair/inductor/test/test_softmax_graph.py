import functools
import unittest
from typing import List, Set
from unittest.mock import patch

import npu_extension_for_inductor
from npu_extension_for_inductor.npu import NPUKernel
import torch
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V


class SoftmaxGraphTest(unittest.TestCase):
    def test_graph_codegen(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        def watch_npu_kernel(kernel, origin, kernels: Set[NPUKernel]):
            if type(kernel) == NPUKernel:
                kernels.add(kernel)
            return origin(kernel)

        cap_kernels = set()
        try:
            with patch.object(V, "set_kernel_handler",
                              functools.partial(watch_npu_kernel, origin=V.set_kernel_handler, kernels=cap_kernels)):
                x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
                test_softmax(x)
        except:
            pass

        cap_kernels = list(cap_kernels)
        self.assertEqual(len(cap_kernels), 1)
        kernel: NPUKernel = cap_kernels[0]
        graph_name = kernel.graph.name
        actual = kernel.graph.codegen(graph_name)
        actual = actual.getvalue().split('\n')
        expect = IndentedBuffer()
        expect.splice(
            f"""def {graph_name}():

                    import os
                    if os.getenv('ASCIR_NOT_READY', None) == "1":
                        return None
                    from pyautofuse import ascir
                    {graph_name} = ascir.HintGraph('{graph_name}')
                    s0 = {graph_name}.create_size("s0")
                    s1 = {graph_name}.create_size("s1")
                    s2 = {graph_name}.create_size("s2")
                    z0 = {graph_name}.create_axis("z0", ascir.SizeExpr([s0,s1]))
                    z1 = {graph_name}.create_axis("z1", ascir.SizeExpr([s2]))
                    size_vars = ascir.ops.Data('size_vars')
                    size_vars.attr.sched.exec_order = 0
                    size_vars.attr.sched.axis = []
                    size_vars.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
                    arg3_1 = ascir.ops.Data('arg3_1')
                    arg3_1.attr.sched.exec_order = 1
                    arg3_1.attr.sched.axis = [z0, z1]
                    arg3_1.y.size = [ascir.SizeExpr([s0,s1,s2])]
                    arg3_1.y.dtype = ascir.dtypes.float16
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
                    max.y.axis = [z0]
                    max.y.strides = [ascir.SizeExpr([])]
                    max.y.size = [ascir.SizeExpr([s0,s1])]
                    store = ascir.ops.Store('store')
                    store.attr.sched.exec_order = 5
                    store.attr.sched.axis = [z0, z1]
                    store.x = max.y
                    store.y.axis = [z0]
                    store.y.strides = [ascir.SizeExpr([])]
                    store.y.size = [ascir.SizeExpr([s0,s1])]
                    buf0 = ascir.ops.Workspace('buf0')
                    buf0.attr.sched.exec_order = 6
                    buf0.attr.sched.axis = [z0, z1]
                    buf0.y.size = [ascir.SizeExpr([]), ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([])]
                    buf0.y.dtype = ascir.dtypes.float32
                    buf0.x = store.y
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
                    load2.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
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
                    buf1.y.size = [ascir.SizeExpr([]), ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
                    buf1.y.dtype = ascir.dtypes.float32
                    buf1.x = store1.y
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
                    sum.y.axis = [z0]
                    sum.y.strides = [ascir.SizeExpr([])]
                    sum.y.size = [ascir.SizeExpr([s0,s1])]
                    store2 = ascir.ops.Store('store2')
                    store2.attr.sched.exec_order = 17
                    store2.attr.sched.axis = [z0, z1]
                    store2.x = sum.y
                    store2.y.axis = [z0]
                    store2.y.strides = [ascir.SizeExpr([])]
                    store2.y.size = [ascir.SizeExpr([s0,s1])]
                    buf2 = ascir.ops.Workspace('buf2')
                    buf2.attr.sched.exec_order = 18
                    buf2.attr.sched.axis = [z0, z1]
                    buf2.y.size = [ascir.SizeExpr([]), ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([])]
                    buf2.y.dtype = ascir.dtypes.float32
                    buf2.x = store2.y
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
                    load5.y.size = [ascir.SizeExpr([s0,s1]), ascir.SizeExpr([s2])]
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
                
                    {graph_name}.set_inputs([size_vars, arg3_1])
                    {graph_name}.set_outputs([buf3])
                    return {graph_name}""")

        expect = expect.getvalue().split('\n')
        self.assertEqual(len(actual), len(expect))
        for i in range(len(actual)):
            self.assertEqual(actual[i].strip(), expect[i].strip())


if __name__ == '__main__':
    unittest.main()
