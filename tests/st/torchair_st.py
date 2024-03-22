import math
import os
os.environ['TNG_LOG_LEVEL'] = '1'
import torchair
import torch
import unittest
import time
import logging

from torchair.core.utils import logger
from torchair.core.backend import TorchNpuGraph
from torchair.ge_concrete_graph.ge_graph import GeGraph
from torchair.ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph, \
    _mapping_assign_op_to_graph_output
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import DataType
from torchair.ge_concrete_graph.graph_pass import optimize_reference_op_redundant_copy
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.backend import initialize_graph_engine
logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = torchair.get_npu_backend(compiler_config=config)


def set_graph_output_dtypes(graph, dtypes):
    _normalize_ge_graph(graph)
    graph.attr["_output_dtypes"].list.i.extend(dtypes)
    graph.attr["_executor_type"].i = ExecutorType.NPU
    input_placements = dict()
    for op in graph.op:
        if op.type == "Data":
            input_placements[op.attr['index'].i] = Placement.HOST if op.output_desc[0].device_type == "CPU" else Placement.DEVICE
    for _, v in sorted(input_placements.items()):
        graph.attr["_input_placements"].list.i.append(v)


class TorchairSt(unittest.TestCase):
    def test_basic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2, 2)
        y = torch.randn(2, 2, 2)
        for i in range(2):
            model(x, y)

    def test_complex_type(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.add(x, x)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2).to(torch.complex32)
        self.assertEqual(model(x).dtype, torch.complex32)
        x = torch.randn(2).to(torch.complex64)
        self.assertEqual(model(x).dtype, torch.complex64)
        x = torch.randn(2).to(torch.complex128)
        self.assertEqual(model(x).dtype, torch.complex128)

    def test_bf16(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2, 2).to(torch.bfloat16)
        y = torch.randn(2, 2, 2).to(torch.bfloat16)
        for i in range(2):
            model(x, y)

    def test_sym_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x, 2)
        model(x, 3)
        model(x, 2.0)
        model(x, 3.0)

    def test_enable_constplaceholder_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        config_cp = CompilerConfig()
        ## TO DO: fix me after ConstPlaceHolder enable
        # config_cp.experimental_config.frozen_parameter = True
        npu_backend_cp = torchair.get_npu_backend(compiler_config=config_cp)
        model = torch.compile(Model(), backend=npu_backend_cp, dynamic=True)
        x = torch.randn(2, 2)
        x = torch.nn.Parameter(x)
        model(x, 2)

    def test_enable_constplaceholder_static(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        config_cp = CompilerConfig()
        ## TO DO: fix me after ConstPlaceHolder enable
        # config_cp.experimental_config.frozen_parameter = True
        npu_backend_cp = torchair.get_npu_backend(compiler_config=config_cp)
        model = torch.compile(Model(), backend=npu_backend_cp, dynamic=False)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        x = torch.nn.Parameter(x)
        model(x, y)

    def test_auto_tune(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        config_auto_tune = CompilerConfig()
        config_auto_tune.aoe_config.aoe_mode = "2"
        config_auto_tune.debug.graph_dump.type = "pbtxt"
        npu_backend_auto_tune = torchair.get_npu_backend(compiler_config=config_auto_tune)

        model = torch.compile(Model(), backend=npu_backend_auto_tune, dynamic=False)
        x = torch.randn(2, 2)
        model(x, 2)

    def test_builtin_with_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                x = torch.add(x, y + z)
                x = torch.add(x, y - z)
                x = torch.add(x, y * z)
                x = torch.add(x, y / z)
                x = torch.add(x, y // z)
                x = torch.add(x, y ** z)
                return x

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x, 2, 3)
        model(x, 3, 4)

    def test_ge_api_support_position_passin_by_kv(self):
        # shape is position input of ge.Empty, check not raise when pass shape by k-v
        ge.Empty(shape=ge.Const(1))

    def test_different_fx_output_from_same_fx_node(self):
        v = torch.ones(2)
        @torch.compile(backend=npu_backend)
        def one_2_two_case1(x):
            return x, x
        x, y = one_2_two_case1(v)
        self.assertTrue(x is y)

        @torch.compile(backend=npu_backend)
        def one_2_two_case2(x):
            return x, x + 1, x
        x, _, y = one_2_two_case2(v)
        self.assertTrue(x is y)

        @torch.compile(backend=npu_backend)
        def one_2_two_case3(x):
            return x + 1, x, x
        _, x, y = one_2_two_case3(v)
        self.assertTrue(x is y)

        @torch.compile(backend=npu_backend)
        def one_2_two_case4(x):
            return x, x, x + 1
        x, y, _ = one_2_two_case4(v)
        self.assertTrue(x is y)

    def test_ge_graph_dump_with_py(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        def get_dumped_py_file_list(dir_path, file_extension='.py'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo_') and i.endswith(f'{file_extension}')]

        for file_name in get_dumped_py_file_list('./'):
            os.remove(os.path.join('./', file_name))

        config = CompilerConfig()
        config.debug.graph_dump.type = "py"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        model = Model()
        model = torch.compile(model, backend=npu_backend)
        x = torch.randn(2, 2)
        output = model(x)

        dumped_py_file_list = get_dumped_py_file_list('./')
        dumped_py_file_list.sort(
            key=lambda file_name: os.path.getmtime(os.path.join('./', file_name)))
        assert dumped_py_file_list.__len__() > 0
        file_name = os.path.join('./', dumped_py_file_list[-1])

        with open(file_name, 'r')as f:
            src = f.read()

        assert src != '# -*- coding: utf-8 -*-\nfrom torch import tensor\n' \
                      'from torchair.ge_concrete_graph import ge_apis as ge\n' \
                      'from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph\n\n'


    def test_1sym_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                a = z.view([x]) + 1.0
                return a

        npu_backend = torchair.get_npu_backend()
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        in4 = torch.randn([3, 2])
        model(6, 3, in4)

    def test_2sym_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                a = z.view([x]) + y.view([x]) + x
                return a

        npu_backend = torchair.get_npu_backend()
        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        in4 = torch.randn([3, 2])
        in3 = torch.randn([2, 3])
        model(6, in3, in4)

    def test_same_sym_pack_merge(self):
        def get_graph_pack_data_num(concrete_graph):
            pack_num = 0
            data_num = 0
            for node in concrete_graph.graph.op:
                if node.type == "Pack":
                    pack_num += 1
                if node.type == "Data":
                    data_num += 1
            return pack_num, data_num

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                pack_num, data_num = get_graph_pack_data_num(args[0])
                assert pack_num == 6, f"before optimize, assert pack op num failed, expect 5, get {pack_num}"
                assert data_num == 5, f"before optimize, assert data op num failed, expect 6, get {data_num}"

                ret = func(*args, **kwargs)

                pack_num, data_num = get_graph_pack_data_num(args[0])
                assert pack_num == 2, f"after optimize, assert pack op num failed, expect 2, get {pack_num}"
                assert data_num == 6, f"after optimize, assert data op num failed, expect 6, get {data_num}"
                return ret

            return wrapper

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z, p, m, n):
                b = p.view([x]) + z.view([x]) + z.view([x])
                c = m.view([x, x]).sum()
                a = torch.stack([n, n, n, n])
                d = (m.view([4, y, y]) + a).sum()
                return b + c - d

        model = Model()

        model = torch.compile(model, backend=npu_backend, dynamic=True)

        z = torch.randn([3, 2])
        p = torch.randn([2, 3])
        m = torch.randn([36])
        n = torch.randn([3, 3])
        model(6, 3, z, p, m, n)

        GeConcreteGraph.__call__ = src_call

    def test_npu_executor_optimize_ref_op_copy(self):
        def get_graph_key_op_num(graph):
            assign_num = 0
            tensormove_num = 0
            netoutput_input_num = 0
            for node in graph.op:
                if node.type == "Assign":
                    assign_num += 1
                elif node.type == "TensorMove":
                    tensormove_num += 1
                elif node.type == "NetOutput":
                    netoutput_input_num = len(node.input)
            return assign_num, tensormove_num, netoutput_input_num

        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            dst = ge.Data(index=0, shape=[3, 1, 16, 8],
                          dtype=DataType.DT_FLOAT, placement='NPU')
            dst1 = ge.Data(index=1, shape=[3, 1, 16, 8],
                           dtype=DataType.DT_FLOAT, placement='NPU')
            src = ge.Data(index=2, shape=[3, 1, 3, 8],
                          dtype=DataType.DT_FLOAT, placement='NPU')
            indices = ge.Data(index=3, shape=[3],
                              dtype=DataType.DT_INT32, placement='NPU')
            dst2 = ge.Data(index=4, shape=[2, 1, 16, 8],
                           dtype=DataType.DT_FLOAT, placement='NPU')
            tm = ge.TensorMove(dst)
            tm1 = ge.TensorMove(dst1)
            dst_ = ge.Scatter(tm, indices, src, reduce="update", axis=-2)
            dst1_ = ge.Scatter(tm1, indices, src, reduce="update", axis=-2)
            assign = ge.Assign(dst, dst_)
            assign_ = ge.Assign(dst1, dst1_)

            tm2 = ge.TensorMove(dst_)
            tm3 = ge.TensorMove(dst1_)
            dst2_ = ge.Scatter(tm2, indices, src, reduce="update", axis=-2)
            dst3_ = ge.Scatter(tm3, indices, src, reduce="update", axis=-2)
            sub = ge.Sub(dst2_, dst3_)
            add = ge.Add(dst_, dst1_)
            sub_squeeze = ge.Squeeze(sub, axis=[1])
            add_squeeze = ge.Squeeze(add, axis=[1])
            sub_tm = ge.TensorMove(sub_squeeze)
            add_tm = ge.TensorMove(add_squeeze)
            dst2_tm = ge.TensorMove(dst2)
            src_list = ge.Transpose(src, [2, 0, 1, 3])
            out1, out2, out3 = ge.ScatterList([sub_tm, add_tm, dst2_tm],
                                              indices, src_list, None, reduce="update", axis=-2)
            assign3 = ge.Assign(dst2, out3)
            assign2 = ge.Assign(sub, out2)
            output = ge.NetOutput([out1, out2, out3])

            set_graph_output_dtypes(graph, [DataType.DT_FLOAT, DataType.DT_FLOAT, DataType.DT_FLOAT])
            executor = TorchNpuGraph()
            optimize_reference_op_redundant_copy(graph)
            assign_num, tm_num, output_in = get_graph_key_op_num(graph)
            assert assign_num == 2, f"after optimize, assert assign op num failed, expect 2, get {assign_num}"
            assert tm_num == 4, f"after optimize, assert TensorMove op num failed, expect4, get {tm_num}"
            assert output_in == 3, f"after optimize, assert output num failed, expect 3, get {output_in}"

            output_ref_input = _mapping_assign_op_to_graph_output(graph)
            executor.load(graph)
            executor.compile()

        dst = torch.ones(3, 1, 16, 8).float().to(npu_device)
        dst1 = torch.ones(3, 1, 16, 8).float().to(npu_device)
        src = torch.randn(3, 1, 3, 8).float().to(npu_device)
        indices = torch.tensor([1, 1]).int().to(npu_device)
        dst2 = torch.ones(3, 1, 16, 8).float().to(npu_device)

        inputs = [dst, dst1, src, indices, dst2]
        assigned_outputs = [None] * len(graph.attr["_output_dtypes"].list.i)
        for output_index, input_index in output_ref_input.items():
            assigned_outputs[output_index] = inputs[input_index]

        outs = executor.run(inputs, assigned_outputs)
        self.assertTrue(len(outs) == 3)
        self.assertTrue(outs[2] is dst2)

    def test_assign_input_in_netoutput(self):
        def _get_graph_output_num(graph):
            netoutput_input_num = 0
            for node in graph.op:
                if node.type == "NetOutput":
                    netoutput_input_num = len(node.input)
            return netoutput_input_num

        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x1 = ge.Data(index=0, shape=[3, 4],
                         dtype=DataType.DT_FLOAT, placement='NPU')
            x2 = ge.Data(index=1, shape=[3, 4],
                         dtype=DataType.DT_FLOAT, placement='NPU')
            res = ge.Add(x1, x2)
            assign = ge.Assign(x1, res)
            res2 = ge.Sub(res, x2)
            output = ge.NetOutput([res, res2])

            set_graph_output_dtypes(graph, [DataType.DT_FLOAT, DataType.DT_FLOAT])
            executor = TorchNpuGraph()
            optimize_reference_op_redundant_copy(graph)
            output_num = _get_graph_output_num(graph)
            assert output_num == 2, f"before optimize, assert output num failed, expect 2, get {output_num}"

            output_ref_input = _mapping_assign_op_to_graph_output(graph)
            output_num = _get_graph_output_num(graph)
            assert output_num == 2, f"after optimize, assert output num failed, expect 2, get {output_num}"

            executor.load(graph)
            executor.compile()

        dst = torch.ones(3, 4).float().to(npu_device)
        dst1 = torch.ones(3, 4).float().to(npu_device)
        inputs = [dst, dst1]
        assigned_outputs = [None] * len(graph.attr["_output_dtypes"].list.i)
        for output_index, input_index in output_ref_input.items():
            assigned_outputs[output_index] = inputs[input_index]

        outs = executor.run(inputs, assigned_outputs)
        self.assertTrue(len(outs) == 2)
        self.assertTrue(outs[0] is dst)

    def test_assign_input_not_netoutput(self):
        def test_assign_input_in_netoutput(self):
            def _get_graph_output_num(graph):
                netoutput_input_num = 0
                for node in graph.op:
                    if node.type == "NetOutput":
                        netoutput_input_num = len(node.input)
                return netoutput_input_num

            initialize_graph_engine()
            from torchair.core import _npu_graph_executor
            import _privateuse1_backend
            npu_device = _privateuse1_backend.npu_device()
            torch.utils.rename_privateuse1_backend("npu")

            with GeGraph() as graph:
                x1 = ge.Data(index=0, shape=[3, 4],
                             dtype=DataType.DT_FLOAT, placement='NPU')
                x2 = ge.Data(index=1, shape=[3, 4],
                             dtype=DataType.DT_FLOAT, placement='NPU')
                res = ge.Add(x1, x2)
                assign = ge.Assign(x1, res)
                output = ge.NetOutput([])

                set_graph_output_dtypes(graph, [DataType.DT_FLOAT])
                executor = TorchNpuGraph()
                optimize_reference_op_redundant_copy(graph)
                output_num = _get_graph_output_num(graph)
                assert output_num == 0, f"before optimize, assert output num failed, expect 0, get {output_num}"

                output_ref_input = _mapping_assign_op_to_graph_output(graph)
                output_num = _get_graph_output_num(graph)
                assert output_num == 1, f"after optimize, assert output num failed, expect 1, get {output_num}"

                executor.load(graph)
                executor.compile()

            dst = torch.ones(3, 4).float().to(npu_device)
            dst1 = torch.ones(3, 4).float().to(npu_device)
            inputs = [dst, dst1]
            assigned_outputs = [None] * len(graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]

            outs = executor.run(inputs, assigned_outputs)
            self.assertTrue(len(outs) == 1)
            self.assertTrue(outs[0] is dst)

    def test_npu_executor_mix_npu_cpu_inputs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32)
            y = torch.ones([], dtype=torch.int32)
            executor.run([x, y])

    def test_static_npu_executor_with_assigned_inputs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu')
            y = torch.ones([], dtype=torch.int32, device='npu')
            z = executor.run([x, y], [x])
            self.assertTrue(z[0] is x)

    def test_dynamic_npu_executor_with_assigned_inputs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu')
            y = torch.ones([], dtype=torch.int32, device='npu')
            z = executor.run([x, y], [x])
            self.assertTrue(z[0] is x)

    def test_input_processing_for_static_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                return x + y + z

        model = torch.compile(Model(), backend=npu_backend, dynamic=False)

        # test nothing to do for input processing
        x0 = torch.randn(2, 4)
        model(x0, x0, x0)
        model(x0, x0, x0)

        # test contiguous for input processing
        x1 = torch.randn(4, 2).t()
        model(x0, x1, x0)
        model(x0, x1, x0)

        # test to_tensor for input processing
        x2 = 1
        model(x0, x1, x2)
        model(x0, x1, x2)

    def test_input_processing_for_dynamic_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, p, x, y, z):
                return x + y + z

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        # test nothing to do for input processing
        x0 = torch.randn(2, 4)
        model(x0, x0, x0, x0)
        model(x0, x0, x0, x0)

        # test contiguous for input processing
        x1 = torch.randn(2, 5)[:, 1:]
        model(x0, x1, x0, x0)
        model(x0, x1, x0, x0)
        x1 = torch.randn(2, 5)[:, :4]
        model(x0, x1, x0, x0)
        model(x0, x1, x0, x0)

        # test to_tensor and eliminate_sym for input processing
        x2 = 1
        model(3, x0, x1, x2)
        model(3, x0, x1, x2)
        x3 = torch.randn(2, 2, 4)[:, 1, :]
        model(x3, x0, x1, x2)
        model(x3, x0, x1, x2)

    def test_output_processing_for_dynamic_graph(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            npu_x = torch.ones([2, 2], dtype=torch.int32).to(npu_device)
            npu_y = torch.ones([], dtype=torch.int32).to(npu_device)
            out = executor.run([npu_x, npu_y])

    def test_dynamic_npu_executor_with_internal_format(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()

        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            cpu_x = torch.ones([2, 2], dtype=torch.int32)
            npu_x = cpu_x.to(npu_device)
            y = torch.ones([], dtype=torch.int32)
            z = executor.run([npu_x, y])
            self.assertTrue(npu_x.device is not y.device)
            z = executor.run([npu_x, y])
            self.assertTrue(npu_x.device is not y.device)

    def test_npu_static_executor(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[100, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

        set_graph_output_dtypes(graph, [DataType.DT_INT32])

        executor = TorchNpuGraph()
        executor.load(graph)
        executor.compile()

        x = torch.ones([1, 2], dtype=torch.int32)
        y = torch.ones([100, 2], dtype=torch.int32)
        result = executor.run((x, y))

        with GeGraph() as graph2:
            x = ge.Data(index=0, shape=[1, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[10, 2],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

        set_graph_output_dtypes(graph2, [DataType.DT_INT32])

        executor2 = TorchNpuGraph()
        executor2.load(graph2)
        executor2.compile()

        x = torch.ones([1, 2], dtype=torch.int32)
        y = torch.ones([10, 2], dtype=torch.int32)
        for i in range(2):
            result = executor2.run((x, y))

    def test_npu_static_executor_with_memory_efficient(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph1:
            a = ge.Data(index=0, shape=[128, 128],
                        dtype=DataType.DT_FLOAT16, placement='CPU')
            b = ge.Data(index=1, shape=[128, 128],
                        dtype=DataType.DT_FLOAT16, placement='CPU')
            c = ge.Add(a, b)
            d = ge.MatMulV2(a, c, bias=None, offset_w=None)
            e = ge.Mul(a, d)
            f = ge.RealDiv(a, e)
            output = ge.NetOutput([f])

        set_graph_output_dtypes(graph1, [DataType.DT_FLOAT16])
        executor = TorchNpuGraph()
        local_options = {}
        local_options["ge.featureBaseRefreshable"] = "1"
        executor.load(graph1, options=local_options)
        executor.compile()

        x = torch.ones([128, 128], dtype=torch.float16)
        y = torch.ones([128, 128], dtype=torch.float16)
        for i in range(3):
            result = executor.run((x, y))


        with GeGraph() as graph2:
            a = ge.Data(index=0, shape=[16, 16],
                        dtype=DataType.DT_FLOAT16, placement='CPU')
            b = ge.Data(index=1, shape=[16, 16],
                        dtype=DataType.DT_FLOAT16, placement='CPU')
            c = ge.Add(a, b)
            d = ge.MatMulV2(a, c, bias=None, offset_w=None)
            e = ge.Mul(a, d)
            f = ge.RealDiv(a, e)
            output = ge.NetOutput([f])

        set_graph_output_dtypes(graph2, [DataType.DT_FLOAT16])
        executor2 = TorchNpuGraph()
        executor2.load(graph2, options=local_options)
        executor2.compile()

        x = torch.ones([16, 16], dtype=torch.float16)
        y = torch.ones([16, 16], dtype=torch.float16)
        for i in range(3):
            result = executor2.run((x, y))

    def test_npu_graph_executor_func(self):
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        in_shape = [2, 3, 4, 5]
        x = torch.ones(in_shape).to(npu_device)
        storage_shape = _npu_graph_executor.GetNpuStorageSizes(x)
        self.assertTrue(storage_shape == in_shape)

    def test_rng_into_graph(self):
        def check_graph(concrete_graph):
            num_data, has_offset, has_seed, has_unpack = 0, False, False, False
            for node in concrete_graph.graph.op:
                if node.type == 'Data':
                    num_data += 1
                if node.type == 'Data' and node.name == 'offset_list':
                    has_offset = True
                if node.type == 'Unpack' and node.name == 'unpack_generator_offsets':
                    assert 'offset_list' in node.input[0]
                    assert node.attr['num'].i == 2
                    has_unpack = True
                if node.type == 'Const' and node.name == 'initial_seed':
                    assert node.attr['_readable_value'].s == b'10'
                    has_seed = True
            logger.debug(f'check_graph index:')
            logger.debug(f'    num_data: {num_data}')
            logger.debug(f'    has_offset: {has_offset}')
            logger.debug(f'    has_unpack: {has_unpack}')
            logger.debug(f'    has_seed: {has_seed}')
            assert num_data == 2 and has_offset and has_seed and has_unpack

        def call_sub(self, *args, **kwargs):
            check_graph(self)
            return args

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = call_sub


        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dp = torch.nn.Dropout(0.3)

            def forward(self, x):
                y = self.dp(x)
                b1 = torch.ops.aten.bernoulli.p(x, 0.8)
                y = y * b1
                return y


        model = Model()
        model = torch.compile(model, backend=npu_backend)
        x = torch.randn(4, 3, 32)
        model(x)

        GeConcreteGraph.__call__ = src_call

    def test_torch_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                assert len(x.size()) >= 1
                a = float(x.size(-1))
                b = 1 / math.sqrt(x.size(-1))
                return a + b

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        for i in range(10, 20):
            x = torch.randn(10, i, i + 1)
            model(x)

    def test_no_broadcast_when_input_output_sym_size_is_equal(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp, size):
                a = torch.ops.aten.expand.default(inp, size)
                b = torch.ops.aten.slice.Tensor(inp)
                return inp + a + b

        def check_graph(concrete_graph):
            num_broadcastto = 0
            num_strideslice = 0
            for node in concrete_graph.graph.op:
                if node.type == 'BroadcastTo':
                    num_broadcastto += 1
                if node.type == 'StridedSlice':
                    num_strideslice += 1

            assert num_broadcastto == 0, f"check number of num_broadcastto {num_broadcastto} == 0 failed"
            assert num_strideslice == 0, f"check number of num_strideslice {num_strideslice} == 0 failed"

        def my_decorator(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                check_graph(args[0])
                return func(*args, **kwargs)
            return wrapper

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = my_decorator(GeConcreteGraph.__call__)

        model = Model()
        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)

        for i in range(10, 15):
            x = torch.randn(i, i + 1, i + 2, i + 3)
            model_dynamic(x, x.size())

        model_static = torch.compile(model, backend=npu_backend, dynamic=False)
        for i in range(10, 15):
            x = torch.randn(i, i + 1, i + 2, i + 3)
            model_static(x, x.size())

        GeConcreteGraph.__call__ = src_call

    def test_remove_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                x = torch.cat([torch.ones(x.size()), torch.ones(y.size())])
                x = torch.ones(x.size())
                x = torch.split(x, z, dim=0)
                return x[-1], x[0]

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        model(torch.randn(2, 2), torch.randn(2, 2), [2, 2])
        model(torch.randn(3, 3), torch.randn(3, 3), [3, 3])
        model(torch.randn(4, 4), torch.randn(4, 4), [4, 4])

    def test_permute_with_no_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp, dims):
                a = torch.ops.aten.permute.default(inp, dims)
                res = torch.ops.aten.add.Scalar(a, 1)
                return res

        def check_graph(concreate_graph):
            num_transpose = 0
            for node in concreate_graph.graph.op:
                if node.type == 'Transpose':
                    num_transpose += 1

            assert num_transpose == 0, f"check number of num_transpose {num_transpose} == 0 failed"

        def my_decorator(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                check_graph(args[0])
                return func(*args, **kwargs)
            return wrapper

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = my_decorator(GeConcreteGraph.__call__)

        model = Model()
        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)

        a = torch.randn(2, 3, 1)
        b = torch.randn(1, 2, 3, 1)
        c = torch.randn(1, 2, 2, 1)
        model_dynamic(a, [2, 0, 1])
        model_dynamic(b, [1, 3, 2, 0])
        model_dynamic(c, [1, 0, 3, 2])
        model_dynamic(c, [1, 0, -1, -2])

        model_static = torch.compile(model, backend=npu_backend, dynamic=False)
        a = torch.randn(2, 3, 1)
        b = torch.randn(1, 2, 3, 1)
        c = torch.randn(1, 2, 2, 1)
        model_static(a, [2, 0, 1])
        model_static(b, [1, 3, 2, 0])
        model_static(c, [1, 0, 3, 2])
        model_static(c, [1, 0, -1, -2])

        GeConcreteGraph.__call__ = src_call

    def test_permute_with_transpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp, dims):
                a = torch.ops.aten.permute.default(inp, dims)
                res = torch.ops.aten.add.Scalar(a, 1)
                return res

        def check_graph(concreate_graph):
            num_transpose = 0
            for node in concreate_graph.graph.op:
                if node.type == 'Transpose':
                    num_transpose += 1

            assert num_transpose != 0, f"check number of num_transpose {num_transpose} != 0 failed"

        def my_decorator(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                check_graph(args[0])
                return func(*args, **kwargs)
            return wrapper

        from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        src_call = GeConcreteGraph.__call__
        GeConcreteGraph.__call__ = my_decorator(GeConcreteGraph.__call__)

        model = Model()
        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)

        a = torch.randn(2, 3, 1)
        b = torch.randn(1, 2, 3, 1)
        c = torch.randn(1, 2, 2, 1)
        model_dynamic(a, [1, 2, 0])
        model_dynamic(b, [2, 3, 1, 0])
        model_dynamic(c, [0, 2, 1, 3])
        model_dynamic(c, [0, -2, 1, -1])

        model_static = torch.compile(model, backend=npu_backend, dynamic=False)
        a = torch.randn(2, 3, 1)
        b = torch.randn(1, 2, 3, 1)
        c = torch.randn(1, 2, 2, 1)
        model_static(a, [1, 2, 0])
        model_static(b, [2, 3, 1, 0])
        model_static(c, [0, 2, 1, 3])
        model_static(c, [0, -2, 1, -1])

        GeConcreteGraph.__call__ = src_call

    def test_set_error_option_path(self):
        config_error = CompilerConfig()
        with self.assertRaises(FileNotFoundError) as context:
            config_error.dump_config.dump_path = "./*****"
        self.assertTrue('Please set legal dir path, ./***** is not found or is not a file directory!'
                        in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            config_error.aoe_config.work_path = "./*****"
        self.assertTrue('Please set legal dir path, ./***** is not found or is not a file directory!'
                        in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            config_error.aoe_config.aoe_config_file = "./*****"
        self.assertTrue('Please set legal file path, ./***** is not found or is not a file!'
                        in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            config_error.fusion_config.fusion_switch_file = "./*****"
        self.assertTrue('Please set legal file path, ./***** is not found or is not a file!'
                        in str(context.exception))

        with self.assertRaises(FileNotFoundError) as context:
            config_error.dump_config.dump_path = None
        self.assertTrue('Please set legal dir path, None is not found or is not a file directory!'
                        in str(context.exception))
        with self.assertRaises(FileNotFoundError) as context:
            config_error.aoe_config.work_path = None
        self.assertTrue('Please set legal dir path, None is not found or is not a file directory!'
                        in str(context.exception))

    def test_set_error_static_model_ops_lower_limit(self):
        config_error1 = CompilerConfig()
        with self.assertRaises(ValueError) as context:
            config_error1.experimental_config.static_model_ops_lower_limit = "-1"
        self.assertTrue("Please set integer type, but got <class 'str'>" in str(context.exception))
        config_error2 = CompilerConfig()
        with self.assertRaises(ValueError) as context:
            config_error2.experimental_config.static_model_ops_lower_limit = -2
        self.assertTrue('Please set value in [-1, 9223372036854775807], -2 is out of range.'
                        in str(context.exception))

    def test_set_option(self):
        if not os.path.exists("./dump"):
            os.mkdir("./dump")
        config_option = CompilerConfig()
        config_option.dump_config.dump_path = "./dump"
        self.assertEqual(config_option.dump_config.dump_path.value, "./dump")
        config_option.aoe_config.work_path = "./dump"
        self.assertEqual(config_option.aoe_config.work_path.value, "./dump")
        config_option.experimental_config.static_model_ops_lower_limit = 0
        self.assertEqual(config_option.experimental_config.static_model_ops_lower_limit.value, '0')

    def test_error_code(self):
        with self.assertRaises(RuntimeError) as context:
            torchair.core.backend.TorchNpuGraph().run(None)
        self.assertTrue('ERR03005 GRAPH internal error' in str(context.exception))        
    
    def test_npu_fx_pass(self):
        fx_pass_config = CompilerConfig()
        fx_pass_config.experimental_config.npu_fx_pass = True
        fx_pass_npu_backend = torchair.get_npu_backend(compiler_config=fx_pass_config)
        
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos_data, sin_data):
            q_embed = (q * cos_data) + (rotate_half(q) * sin_data)
            k_embed = (k * cos_data) + (rotate_half(k) * sin_data)
            return q_embed, k_embed

        compiled_fn = torch.compile(apply_rotary_pos_emb, backend=fx_pass_npu_backend)
        compiled_fn(
            torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16), 
            torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16)
            )

    def test_viewofoutput_dynamic_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[0:2, 0:2].transpose(0, 1)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input1), True)

    def test_viewofoutput_dynamic_symexpr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:4, 1:4].transpose(0, 1)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input1), True)

    def test_viewofoutput_dynamic_sym_from_different_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, a):
                return torch.split(x[1:4, 1:4].transpose(0, 1), a)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[0], input1), True)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[1], input1), True)

    def test_viewofoutput_dynamic_input_changed(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, a):
                return torch.split(x[1:4, 1:4].transpose(0, 1), a)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[0], input1), True)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[1], input1), True)
        input1 = torch.randn(6, 6).float()
        res_dynamo = model_dynamo(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[0], input1), True)
        self.assertEqual(torch._C._is_alias_of(res_dynamo[1], input1), True)

    def test_viewofoutput_static(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:2, 1:3].transpose(0, 1)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=False)

        input1 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input1), True)

    def test_without_viewofoutput1(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x[0:2, 0:1].transpose(0, 1) + y[0:2, 0:1].transpose(0, 1)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        input2 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1, input2)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input1), False)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input2), False)

    def test_without_viewofoutput2(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x[0:2, 0:1].transpose(0, 1) + y[0:2, 0:1].transpose(0, 1)
                return a[0:1, 0:1].transpose(0, 1)

        model_dynamo = torch.compile(Model(), backend=npu_backend, dynamic=True)
        model_eager = Model()

        input1 = torch.randn(4, 4).float()
        input2 = torch.randn(4, 4).float()
        res_dynamo = model_dynamo(input1, input2)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input1), False)
        self.assertEqual(torch._C._is_alias_of(res_dynamo, input2), False)


if __name__ == '__main__':
    unittest.main()
