import math
import os
import shutil
import sys
import contextlib

os.environ['TNG_LOG_LEVEL'] = '0'
import torchair
import torch
import unittest
import time
import logging

from torchair.core.utils import logger
from torchair.core._backend import TorchNpuGraph
from torchair.ge._ge_graph import GeGraph, Const, _ge_dtype_to_ge_proto_dtype
from torchair._ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph, \
    _mapping_assign_op_to_graph_output, replace_data_to_refdata, GeConcreteGraph
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.ge._ge_graph import DataType
from torchair._ge_concrete_graph.graph_pass import optimize_reference_op_redundant_copy
from torchair.configs.compiler_config import CompilerConfig
from torchair.core._backend import initialize_graph_engine
from torchair_st_utils import capture_stdout

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
            input_placements[op.attr['index'].i] = Placement.HOST if op.output_desc[
                                                                         0].device_type == "CPU" else Placement.DEVICE
    for _, v in sorted(input_placements.items()):
        graph.attr["_input_placements"].list.i.append(v)


def register_is_npu():
    @property
    def _is_npu(self):
        return not self.is_cpu

    torch.Tensor.is_npu = _is_npu


@contextlib.contextmanager
def set_env_var(key, value):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        self.call_bak = GeConcreteGraph.__call__
        self.optimize_bak = GeConcreteGraph.optimize_graph_without_runtime
        torchair.core._backend._GLOBAL_COMPILE_OPTION = None
        return super().setUp()

    def tearDown(self) -> None:
        GeConcreteGraph.__call__ = self.call_bak
        GeConcreteGraph.optimize_graph_without_runtime = self.optimize_bak
        return super().tearDown()

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

    def test_multiple_input_types(self):
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
        x = torch.randn(2, 2, 2).to(torch.bfloat16)
        self.assertEqual(model(x).dtype, torch.bfloat16)

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

    def test_fx_dumper(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        dumper_config = CompilerConfig()
        dumper_config.debug.data_dump.type = "npy"
        dumper_backend = torchair.get_npu_backend(compiler_config=dumper_config)

        model = torch.compile(Model(), backend=dumper_backend)
        x = torch.randn([2, 3, 4, 5])
        model(x)
        y = torch.randn([2, 3, 4, 7])
        model(y)

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

        with open(file_name, 'r') as f:
            src = f.read()

        assert src != '# -*- coding: utf-8 -*-\nfrom torch import tensor\n' \
                      'from torchair._ge_concrete_graph import ge_apis as ge\n' \
                      'from torchair.ge._ge_graph import get_default_ge_graph\n\n'

    def test_sym_pack(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                a = z.view([x]) + y.view([x, 1]) + x
                return a

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
                assert pack_num == 10, f"before optimize, assert pack op num failed, expect 10, get {pack_num}"
                assert data_num == 8, f"before optimize, assert data op num failed, expect 8, get {data_num}"

                ret = func(*args, **kwargs)

                pack_num, data_num = get_graph_pack_data_num(args[0])
                assert pack_num == 2, f"after optimize, assert pack op num failed, expect 2, get {pack_num}"
                assert data_num == 6, f"after optimize, assert data op num failed, expect 6, get {data_num}"
                return ret

            return wrapper

        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)

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

        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization

    def test_npu_executor_optimize_ref_op_copy(self):
        def get_graph_key_op_num(graph):
            netoutput_input_num = 0
            node_count_dict = {"Assign": 0, "TensorMove": 0, "Data": 0, "RefData": 0}
            for node in graph.op:
                if node.type == "NetOutput":
                    netoutput_input_num = len(node.input)
                elif node.type in node_count_dict.keys():
                    node_count_dict[node.type] += 1
            return node_count_dict, netoutput_input_num

        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            dst = ge.Data(index=0, shape=[3, 1, 16, 8], dtype=DataType.DT_FLOAT, placement='NPU')
            dst1 = ge.Data(index=1, shape=[3, 1, 16, 8], dtype=DataType.DT_FLOAT, placement='NPU')
            src = ge.Data(index=2, shape=[3, 1, 3, 8], dtype=DataType.DT_FLOAT, placement='NPU')
            indices = ge.Data(index=3, shape=[3], dtype=DataType.DT_INT32, placement='NPU')
            dst2 = ge.Data(index=4, shape=[2, 1, 16, 8], dtype=DataType.DT_FLOAT, placement='NPU')
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
            ref_data_idx = optimize_reference_op_redundant_copy(graph)

            node_count_dict, output_in = get_graph_key_op_num(graph)
            assert node_count_dict["Assign"] == 2, \
                f'after optimize, assert assign op num failed, expect 2, get {node_count_dict["Assign"]}'
            assert node_count_dict["TensorMove"] == 4, \
                f'after optimize, assert TensorMove op num failed, expect 4, get {node_count_dict["TensorMove"]}'
            assert output_in == 3, f'after optimize, assert output num failed, expect 3, get {output_in}'
            assert node_count_dict["Data"] == 5, \
                f'after optimize, assert input data num failed, expect 5, get {node_count_dict["Data"]}'
            assert node_count_dict["RefData"] == 0, \
                f'after optimize, assert output num failed, expect 0, get {node_count_dict["RefData"]}'

            output_ref_input = _mapping_assign_op_to_graph_output(graph)
            _, output_in = get_graph_key_op_num(graph)
            assert output_in == 4, f"after optimize, assert output num failed, expect 4, get {output_in}"

            dst = torch.ones(3, 1, 16, 8).float().to(npu_device)
            dst1 = torch.ones(3, 1, 16, 8).float().to(npu_device)
            src = torch.randn(3, 1, 3, 8).float().to(npu_device)
            indices = torch.tensor([1, 1]).int().to(npu_device)
            dst2 = torch.ones(3, 1, 16, 8).float().to(npu_device)
            inputs = [dst, dst1, src, indices, dst2]

            all_ref_data_idx = set()
            for idx in ref_data_idx:
                all_ref_data_idx.add(idx)
            for k, v in output_ref_input.items():
                all_ref_data_idx.add(v)

            replace_data_to_refdata(graph, all_ref_data_idx, inputs)
            node_count_dict, output_in = get_graph_key_op_num(graph)
            assert node_count_dict["Assign"] == 1, \
                f'after optimize, assert assign op num failed, expect 1, get {node_count_dict["Assign"]}'
            assert node_count_dict["Data"] == 2, \
                f'after optimize, assert input data num failed, expect 3, get {node_count_dict["Data"]}'
            assert output_in == 4, f'after optimize, assert output num failed, expect 4, get {output_in}'
            assert node_count_dict["RefData"] == 3, \
                f'after optimize, assert output num failed, expect 3, get {node_count_dict["RefData"]}'

            assigned_outputs = [None] * len(graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]

            executor.load(graph)
            executor.compile()

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
        _privateuse1_backend.register_hook()


        with GeGraph() as graph:
            x1 = ge.Data(index=0, shape=[3, 4], dtype=DataType.DT_FLOAT, placement='NPU')
            x2 = ge.Data(index=1, shape=[3, 4], dtype=DataType.DT_FLOAT, placement='NPU')
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
                x1 = ge.Data(index=0, shape=[3, 4], dtype=DataType.DT_FLOAT, placement='NPU')
                x2 = ge.Data(index=1, shape=[3, 4], dtype=DataType.DT_FLOAT, placement='NPU')
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
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='CPU')
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
            x = ge.Data(index=0, shape=[1, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_FLOAT, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_FLOAT])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.float, device='npu')
            y = torch.ones([], dtype=torch.float, device='npu')
            z = executor.run([x, y], [x])
            k = executor.run([x, y], [x])
            self.assertTrue(z[0] is x)
            self.assertTrue(k[0] is x)

    def test_dynamic_npu_executor_with_assigned_inputs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='CPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu')
            y = torch.ones([2, ], dtype=torch.int32, device='npu')
            z = executor.run([x, y], [x])
            x1 = torch.ones([3, 3], dtype=torch.int32, device='npu')
            y1 = torch.ones([3, ], dtype=torch.int32, device='npu')
            z1 = executor.run([x1, y1], [x1])
            self.assertTrue(z[0] is x)
            self.assertTrue(z1[0] is x1)

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
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='NPU')
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
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='CPU')
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
            x = ge.Data(index=0, shape=[1, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            y = ge.Data(index=1, shape=[100, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

        set_graph_output_dtypes(graph, [DataType.DT_FLOAT])

        executor = TorchNpuGraph()
        executor.load(graph)
        executor.compile()

        x = torch.ones([1, 2], dtype=torch.float)
        y = torch.ones([100, 2], dtype=torch.float)
        result = executor.run((x, y))

        with GeGraph() as graph2:
            x = ge.Data(index=0, shape=[1, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            y = ge.Data(index=1, shape=[10, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

        set_graph_output_dtypes(graph2, [DataType.DT_FLOAT])

        executor2 = TorchNpuGraph()
        executor2.load(graph2)
        executor2.compile()

        x = torch.ones([1, 2], dtype=torch.float)
        y = torch.ones([10, 2], dtype=torch.float)
        for i in range(2):
            result = executor2.run((x, y))

    def test_npu_static_executor_with_memory_efficient(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph1:
            a = ge.Data(index=0, shape=[128, 128], dtype=DataType.DT_FLOAT, placement='CPU')
            b = ge.Data(index=1, shape=[128, 128], dtype=DataType.DT_FLOAT, placement='CPU')
            c = ge.Add(a, b)
            d = ge.MatMulV2(a, c, bias=None, offset_w=None)
            e = ge.Mul(a, d)
            f = ge.RealDiv(a, e)
            output = ge.NetOutput([f])

        set_graph_output_dtypes(graph1, [DataType.DT_FLOAT])
        executor = TorchNpuGraph()
        local_options = {}
        local_options["ge.featureBaseRefreshable"] = "1"
        executor.load(graph1, options=local_options)
        executor.compile()

        x = torch.ones([128, 128], dtype=torch.float)
        y = torch.ones([128, 128], dtype=torch.float)
        for i in range(3):
            result = executor.run((x, y))

        with GeGraph() as graph2:
            a = ge.Data(index=0, shape=[16, 16], dtype=DataType.DT_FLOAT, placement='CPU')
            b = ge.Data(index=1, shape=[16, 16], dtype=DataType.DT_FLOAT, placement='CPU')
            c = ge.Add(a, b)
            d = ge.MatMulV2(a, c, bias=None, offset_w=None)
            e = ge.Mul(a, d)
            f = ge.RealDiv(a, e)
            output = ge.NetOutput([f])

        set_graph_output_dtypes(graph2, [DataType.DT_FLOAT])
        executor2 = TorchNpuGraph()
        executor2.load(graph2, options=local_options)
        executor2.compile()

        x = torch.ones([16, 16], dtype=torch.float)
        y = torch.ones([16, 16], dtype=torch.float)
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
                    has_seed = True
            logger.debug(f'check_graph index:')
            logger.debug(f'    num_data: {num_data}')
            logger.debug(f'    has_offset: {has_offset}')
            logger.debug(f'    has_unpack: {has_unpack}')
            logger.debug(f'    has_seed: {has_seed}')
            assert num_data == 2 and has_offset and has_seed and has_unpack

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                check_graph(args[0])
                return call(*args, **kwargs)

            return wrapper

        GeConcreteGraph.__call__ = decorator(GeConcreteGraph.__call__)

        import _privateuse1_backend
        _privateuse1_backend.register_generator()
        src_gen = torch.default_generator
        torch.default_generator = _privateuse1_backend.default_generator(0)

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
        torch.default_generator = src_gen

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
            torchair.core._backend.TorchNpuGraph().run(None)
        self.assertTrue('ERR03005 GRAPH internal error' in str(context.exception))

    def test_npu_fx_pass(self):
        fx_pass_config = CompilerConfig()
        fx_pass_config.experimental_config.npu_fx_pass = True
        fx_pass_npu_backend = torchair.get_npu_backend(compiler_config=fx_pass_config)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
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

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res = model(input1)
        self.assertEqual(torch._C._is_alias_of(res, input1), True)
        self.assertEqual(res.size(), torch.Size([2, 2]))
        self.assertEqual(res.stride(), (1, 4))
        self.assertEqual(res.storage_offset(), 0)

    def test_viewofoutput_dynamic_symexpr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:4, 1:4].transpose(0, 1)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res = model(input1)
        self.assertEqual(torch._C._is_alias_of(res, input1), True)
        self.assertEqual(res.size(), torch.Size([3, 3]))
        self.assertEqual(res.stride(), (1, 4))
        self.assertEqual(res.storage_offset(), 5)

    def test_viewofoutput_dynamic_sym_from_different_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, a):
                return torch.split(x[1:4, 1:4].transpose(0, 1), a)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res = model(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res[0], input1), True)
        self.assertEqual(res[0].size(), torch.Size([2, 3]))
        self.assertEqual(res[0].stride(), (1, 4))
        self.assertEqual(res[0].storage_offset(), 5)
        self.assertEqual(torch._C._is_alias_of(res[1], input1), True)
        self.assertEqual(res[1].size(), torch.Size([1, 3]))
        self.assertEqual(res[1].stride(), (1, 4))
        self.assertEqual(res[1].storage_offset(), 7)

    def test_viewofoutput_dynamic_input_changed(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, a):
                return torch.split(x[1:4, 1:4].transpose(0, 1), a)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        res = model(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res[0], input1), True)
        self.assertEqual(torch._C._is_alias_of(res[1], input1), True)
        input1 = torch.randn(6, 6).float()
        res = model(input1, 2)
        self.assertEqual(torch._C._is_alias_of(res[0], input1), True)
        self.assertEqual(res[0].size(), torch.Size([2, 3]))
        self.assertEqual(res[0].stride(), (1, 6))
        self.assertEqual(res[0].storage_offset(), 7)
        self.assertEqual(torch._C._is_alias_of(res[1], input1), True)
        self.assertEqual(res[1].size(), torch.Size([1, 3]))
        self.assertEqual(res[1].stride(), (1, 6))
        self.assertEqual(res[1].storage_offset(), 9)

    def test_viewofoutput_static(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:2, 1:3].transpose(0, 1)

        model = torch.compile(Model(), backend=npu_backend, dynamic=False)

        input1 = torch.randn(4, 4).float()
        res = model(input1)
        self.assertEqual(torch._C._is_alias_of(res, input1), True)
        self.assertEqual(res.size(), torch.Size([2, 1]))
        self.assertEqual(res.stride(), (1, 4))
        self.assertEqual(res.storage_offset(), 5)

    def test_without_viewofoutput1(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x[0:2, 0:1].transpose(0, 1) + y[0:2, 0:1].transpose(0, 1)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        input2 = torch.randn(4, 4).float()
        res = model(input1, input2)
        self.assertEqual(torch._C._is_alias_of(res, input1), False)
        self.assertEqual(torch._C._is_alias_of(res, input2), False)

    def test_without_viewofoutput2(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x[0:2, 0:1].transpose(0, 1) + y[0:2, 0:1].transpose(0, 1)
                return a[0:1, 0:1].transpose(0, 1)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)

        input1 = torch.randn(4, 4).float()
        input2 = torch.randn(4, 4).float()
        res = model(input1, input2)
        self.assertEqual(torch._C._is_alias_of(res, input1), False)
        self.assertEqual(torch._C._is_alias_of(res, input2), False)

    def test_topk(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, b):
                torch.topk(x, b)

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        x = torch.randn([4])
        b = 3
        model(x, b)
        x = torch.randn([6, 7])
        b = 4
        model(x, b)

    def test_autograd_sym(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + 1
                return x, x.size()

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        input0 = torch.randn(size=(4, 2, 4, 4), dtype=torch.float32, requires_grad=True)
        res = model(input0)
        self.assertEqual(res[1], torch.Size([4, 2, 4, 4]))

    def test_autograd_symexpr(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                assert len(x.size()) >= 1
                a = float(x.size(-1))
                b = math.sqrt(x.size(-1))
                c = b / b
                return a + c

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        x = torch.randn(10, 1, 2)
        res = model(x)
        self.assertEqual(res, 3)

    def test_ge_const(self):
        inputs = torch.randn(20, 16, 50, dtype=torch.float)
        scale, zero_point = 1.0, 0
        qint8 = torch.quantize_per_tensor(inputs, scale, zero_point, torch.qint8)
        quint8 = torch.quantize_per_tensor(inputs, scale, zero_point, torch.quint8)
        qint32 = torch.quantize_per_tensor(inputs, scale, zero_point, torch.qint32)
        res = Const(qint8)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_QINT8), res.desc.dtype)
        res = Const(quint8)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_QUINT8), res.desc.dtype)
        res = Const(qint32)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_QINT32), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.float16)
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT16), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.bfloat16)
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_BF16), res.desc.dtype)

        v = torch.randn(0, 3).to(torch.complex32)
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_COMPLEX32), res.desc.dtype)

        v = torch.randn(0, 3).to(torch.complex64)
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_COMPLEX64), res.desc.dtype)

        v = 1
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_INT64), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.float16)
        v = v.numpy()
        res = Const(v)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT16), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.float32)
        res = Const(v, dtype=DataType.DT_FLOAT16)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT16), res.desc.dtype)

        v = torch.tensor([1]).to(torch.float16)
        res = Const(v, dtype=DataType.DT_BF16)
        self.assertEqual(res.node.attr['_readable_value'].s, b'tensor([1.], dtype=torch.float16)')
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_BF16), res.desc.dtype)

        v = 1
        res = Const(v, dtype=DataType.DT_FLOAT16)
        self.assertEqual(res.node.attr['_readable_value'].s, b'1')
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT16), res.desc.dtype)

        v = torch.tensor([1]).to(torch.float32)
        v = v.numpy()
        res = Const(v, dtype=DataType.DT_FLOAT16)
        self.assertEqual(res.node.attr['_readable_value'].s, b'array([1.], dtype=float32)')
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT16), res.desc.dtype)

        v = 1
        res = Const(v, dtype=DataType.DT_BF16)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_BF16), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.float32)
        v = v.numpy()
        res = Const(v, dtype=DataType.DT_BF16)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_BF16), res.desc.dtype)

        v = torch.randn(2, 3).to(torch.float32)
        v = v.numpy()
        res = Const(v, dtype=DataType.DT_COMPLEX32)
        self.assertEqual(_ge_dtype_to_ge_proto_dtype(DataType.DT_COMPLEX32), res.desc.dtype)

    def test_view_operator_optimize(self):
        def get_graph_transpose_reshape_num(concrete_graph):
            transpose_num = 0
            reshape_num = 0
            for node in concrete_graph.graph.op:
                if node.type == "Transpose":
                    transpose_num += 1
                if node.type == "Reshape":
                    reshape_num += 1
            return transpose_num, reshape_num

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                ret = func(*args, **kwargs)
                transpose_num, reshape_num = get_graph_transpose_reshape_num(args[0])
                assert transpose_num == 1, f"assert transpose op num failed, expect 1, get {transpose_num}"
                assert reshape_num == 1, f"assert reshape op num failed, expect 1, get {reshape_num}"
                return ret

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v = x.view(6, 6)
                t = v.transpose(0, 1)
                v2 = t.view(3, 2, 2, 3)
                t2 = v2.transpose(1, 3)
                res = t2 + 1
                return res

        model = Model()
        config_view = CompilerConfig()
        config_view.experimental_config.enable_view_optimize = True
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)

        x = torch.randn([36])
        model(x)

    def test_view_operator_optimize_to_reshape(self):
        def get_graph_transpose_reshape_num(concrete_graph):
            transpose_num = 0
            reshape_num = 0
            for node in concrete_graph.graph.op:
                if node.type == "Transpose":
                    transpose_num += 1
                if node.type == "Reshape":
                    reshape_num += 1
            return transpose_num, reshape_num

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                ret = func(*args, **kwargs)
                transpose_num, reshape_num = get_graph_transpose_reshape_num(args[0])
                assert transpose_num == 0, f"assert transpose op num failed, expect 0, get {transpose_num}"
                assert reshape_num == 1, f"assert reshape op num failed, expect 1, get {reshape_num}"
                return ret

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, a):
                v1 = x.view(a, 32, 1, 128)
                t1 = v1.permute(0, 2, 1, 3)
                v2 = t1.view(a, 1, 4096)
                v3 = v2.view(a, 4096)
                res = v3 + 1
                return res

        model = Model()
        config_view = CompilerConfig()
        config_view.experimental_config.enable_view_optimize = True
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=True)

        x = torch.randn([256, 1, 128])
        a = 8
        model(x, a)

    def test_view_operator_repeat_gather(self):
        def get_graph_gather_reshape_num(concrete_graph):
            gather_num = 0
            transpose_num = 0
            reshape_num = 0
            for node in concrete_graph.graph.op:
                if node.type == "Gather":
                    gather_num += 1
                if node.type == "Transpose":
                    transpose_num += 1
                if node.type == "Reshape":
                    reshape_num += 1
            return gather_num, transpose_num, reshape_num

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                ret = func(*args, **kwargs)
                gather_num, transpose_num, reshape_num = get_graph_gather_reshape_num(args[0])
                assert gather_num == 2, f"assert gather op num failed, expect 2, get {gather_num}"
                assert transpose_num == 0, f"assert transpose op num failed, expect 0, get {transpose_num}"
                assert reshape_num == 1, f"assert reshape op num failed, expect 1, get {reshape_num}"
                return ret

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                t1 = x.transpose(2, 3)
                res = t1 + 1
                return res

        model = Model()
        config_view = CompilerConfig()
        config_view.experimental_config.enable_view_optimize = True
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=True)

        x = torch.randn([6, 6, 4, 1])
        model(x)

    def test_view_operator_repeat_transpose(self):
        def get_graph_gather_reshape_num(concrete_graph):
            gather_num = 0
            transpose_num = 0
            reshape_num = 0
            for node in concrete_graph.graph.op:
                if node.type == "Gather":
                    gather_num += 1
                if node.type == "Transpose":
                    transpose_num += 1
                if node.type == "Reshape":
                    reshape_num += 1
            return gather_num, transpose_num, reshape_num

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                ret = func(*args, **kwargs)
                gather_num, transpose_num, reshape_num = get_graph_gather_reshape_num(args[0])
                assert gather_num == 2, f"assert gather op num failed, expect 2, get {gather_num}"
                assert transpose_num == 1, f"assert transpose op num failed, expect 1, get {transpose_num}"
                assert reshape_num == 2, f"assert reshape op num failed, expect 2, get {reshape_num}"
                return ret

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                t1 = x.transpose(0, 2)
                add = t1 + 1
                sub = t1 - 1
                res = add + sub
                return res

        model = Model()
        config_view = CompilerConfig()
        config_view.experimental_config.enable_view_optimize = True
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=True)

        x = torch.randn([3, 1, 4])
        model(x)

    def test_create_torch_tensor_success(self):
        torchair.llm_datadist.create_npu_tensors([0], torch.float, [0])

    def test_create_torch_tensor_check_failed(self):
        with self.assertRaises(TypeError):
            torchair.llm_datadist.create_npu_tensors(["str"], torch.float, [0])
        with self.assertRaises(TypeError):
            torchair.llm_datadist.create_npu_tensors([1], "str", [0])
        with self.assertRaises(TypeError):
            torchair.llm_datadist.create_npu_tensors([1], torch.float, ["str"])

    def test_torch_floor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                a = x.size(dim=0) + 1.0
                b = math.floor(a)
                res = b * 3
                return res

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        x = torch.randn([10, 2])
        res = model(x)
        self.assertEqual(res, 33)

    def test_squeeze_opt_for_dim_not_one(self):
        def check_graph_key_op_num(concrete_graph):
            num_identity = 0
            num_squeeze = 0
            for node in concrete_graph.graph.op:
                if node.type == 'Identity':
                    num_identity += 1
                if node.type == 'Squeeze':
                    num_squeeze += 1
            assert num_identity == 2, f"check number of Identity {num_identity} == 2 failed"
            assert num_squeeze == 3, f"check number of Squeeze {num_squeeze} == 3 failed"

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                check_graph_key_op_num(args[0])
                return func(*args, **kwargs)

            return wrapper

        GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, l, m, n):
                y = x.squeeze(1) + 1  # to squeeze
                z = x.squeeze(2) + 1  # to identity
                l = l.squeeze() + 1  # to identity
                m = m.squeeze() + 1  # to squeeze
                n = n.squeeze() + 1  # to squeeze

                return y, z, l, m, n

        in1 = torch.randn([2, 1, 3])
        in2 = torch.randn([2, 3, 4])
        in3 = torch.randn([2, 1, 1])
        in4 = torch.randn([1, 1, 1])
        model = Model()

        model_dynamic = torch.compile(model, backend=npu_backend, dynamic=True)
        model_dynamic(in1, in2, in3, in4)

        model_static = torch.compile(model, backend=npu_backend, dynamic=False)
        model_static(in1, in2, in3, in4)

    def test_recompile_of_symsize(self):
        torch._dynamo.config.error_on_recompile = True

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                view = x.reshape(-1, 8192)
                permute = y.permute(1, 0)
                mm = torch.mm(view, permute)
                view2 = mm.view(x.size(0), x.size(1), 3072)
                return view2

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)

        x = torch.randn([4, 190, 8192])
        y = torch.randn([3072, 8192])
        model(x, y)

        x = torch.randn([4, 180, 8192])
        y = torch.randn([3072, 8192])
        model(x, y)

        torch._dynamo.config.error_on_recompile = False

    def test_fx_data_dump_step(self):
        config.dump_config.enable_dump = True
        config.dump_config.dump_step = "0"
        config.dump_config.dump_step = "12"
        config.dump_config.dump_step = "02|3|4"
        config.dump_config.dump_step = "0-1"
        config.dump_config.dump_step = "0|1|2-5|6"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "0ad"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "0|2&"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "--"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "1-"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "-1"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "02||||34"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "1--"
        with self.assertRaises(ValueError):
            config.dump_config.dump_step = "1--6"

    def test_fx_data_dump_layer(self):
        config.dump_config.enable_dump = True
        config.dump_config.dump_layer = "Add_1Mul_1/Square\Add_2Add_3.Add_4 Add5"
        config.dump_config.dump_layer = "Add_1Mul_1 Add5"
        with self.assertRaises(ValueError):
            config.dump_config.dump_layer = "Add|"

    def test_frozen_input_static(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend

        with GeGraph() as graph1:
            a = ge.Data(index=0, shape=[128, 128], dtype=DataType.DT_FLOAT, placement='CPU')
            b = ge.Data(index=1, shape=[1, 2], dtype=DataType.DT_FLOAT, placement='CPU')
            d = ge.Add(a, b)
            output = ge.NetOutput([d])

        set_graph_output_dtypes(graph1, [DataType.DT_FLOAT])
        executor = TorchNpuGraph()
        executor.load(graph1, options={"frozenInput": "0,1"})
        executor.compile()

        x = torch.ones([128, 128], dtype=torch.float)
        y = torch.ones([1, 2], dtype=torch.float)
        for i in range(2):
            result = executor.run((x, y))

    def test_frozen_input_dynamic(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"frozenInput": "0,1"})
            executor.compile()

            npu_x = torch.ones([2, 2], dtype=torch.int32).to(npu_device)
            npu_y = torch.ones([], dtype=torch.int32).to(npu_device)
            for i in range(2):
                out = executor.run([npu_x, npu_y])

    def test_frozen_input_no_used(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[-1, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"frozenInput": "0,0"})
            executor.compile()

            npu_x = torch.ones([2, 2], dtype=torch.int32).to(npu_device)
            npu_y = torch.ones([], dtype=torch.int32).to(npu_device)
            for i in range(2):
                out = executor.run([npu_x, npu_y])


    def test_as_numpy(self):
        from torchair.fx_dumper import _as_numpy
        import numpy as np
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        np_array = _as_numpy(x)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertEqual(np_array.dtype, expected.dtype)

    def test_dynamic_npu_executor_with_reuse_input_addrs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph, set_env_var("ST_OUTPUT_REUSE_INPUT_ADDR", ""):
            copy = ge.Data(index=0, shape=[1, 1, -1, -1], dtype=DataType.DT_INT32, placement='NPU')
            indices = ge.Data(index=1, shape=[1], dtype=DataType.DT_INT32, placement='NPU')
            updates = ge.Data(index=2, shape=[1, 1, 1, -1], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Scatter(copy, indices, updates, reduce="update", axis=0)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.compile()

            copy = torch.ones([1, 1, 2, 8], dtype=torch.int32).to(npu_device)
            indices = torch.ones([1], dtype=torch.int32).to(npu_device)
            updates = torch.ones([1, 1, 1, 8], dtype=torch.int32).to(npu_device)
            z = executor.run([copy, indices, updates])
            self.assertTrue(z[0].data_ptr() == copy.data_ptr())

    def test_directory_generation(self):
        import re

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = Model()
        test_config = torchair.CompilerConfig()
        test_config.debug.graph_dump.type = "pbtxt"
        test_config.debug.graph_dump.path = "./test_directory_generation/dir"
        if os.path.exists(test_config.debug.graph_dump.path):
            shutil.rmtree(test_config.debug.graph_dump.path)
        test_config.ge_config.export_compile_stat = "0"
        test_npu_backend = torchair.get_npu_backend(compiler_config=test_config)
        test_model = torch.compile(model, backend=test_npu_backend)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        test_model(x, y)
        test_type = test_config.debug.graph_dump.type.value
        path = os.path.realpath(os.path.dirname(test_config.debug.graph_dump.path))
        path = os.path.realpath(test_config.debug.graph_dump.path)
        self.assertTrue(os.path.isdir(path), f"directory {path} does not exist.")
        test_files = [f for f in os.listdir(path) if f.endswith(".pbtxt")]
        self.assertEqual(len(test_files), 2, f"found {test_type} files in {path}")
        info = []
        for f in test_files:
            match = re.match(r"dynamo_(optimized|original)_graph_(\d+)_rank_(\d+)_pid_(\d+)_.*", f)
            self.assertIsNotNone(match, f"Filename {f} does not match expected pattern")
            info.append(match.groups())
        (type1, gid1, rankid1, pid1), (type2, gid2, rankid2, pid2) = info
        self.assertIn({type1, type2}, [{"optimized", "original"}], "Both files must be one optimized and one original")
        self.assertEqual(gid1, gid2, "Mismatched graph_id between files")
        self.assertEqual(rankid1, rankid2, "Mismatched rank_id between files")
        self.assertEqual(pid1, pid2, "Mismatched pid between files")

    def test_fx_and_ge_shape_not_same(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")
        from torchair.ge._ge_graph import Tensor
        from torchair._ge_concrete_graph.ge_ir_pb2 import OpDef, TensorDescriptor

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            z.set_meta(torch.ones([2, 2]))
            output = ge.NetOutput([z])            
            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            node = OpDef()
            node.name = "node1"
            node.output_desc.append(TensorDescriptor())

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.set_hint_shape([[2, 2], [2, 2]], [[1, 2]])
            with self.assertRaises(RuntimeError) as context:
                executor.compile()
                self.assertTrue('The dim of Ascend net output: [2, 2] '
                'is not equal to FX net output: [1, 2]' in context.exception)

    def test_fx_and_ge_shape_num_same(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")
        from torchair.ge._ge_graph import Tensor
        from torchair._ge_concrete_graph.ge_ir_pb2 import OpDef, TensorDescriptor

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            z.set_meta(torch.ones([2, 2]))
            output = ge.NetOutput([z])            
            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            node = OpDef()
            node.name = "node1"
            node.output_desc.append(TensorDescriptor())

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.set_hint_shape([[2, 2], [2, 2]], [[2, 2], [2, 2]])
            with self.assertRaises(RuntimeError) as context:
                executor.compile()
                self.assertTrue('The number of Ascend net output: 1 '
                'is not equal to FX net outputs: 2' in context.exception)  

    def test_fx_and_ge_shape_size_not_same(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[2, 2], dtype=DataType.DT_INT32, placement='NPU')
            z = ge.Add(x, y)
            z.set_meta(torch.ones([2, 2]))
            output = ge.NetOutput([z])
            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph)
            executor.set_hint_shape([[2, 2], [2, 2]], [[1, 2, 3]])
            with self.assertRaises(RuntimeError) as context:
                executor.compile()
                self.assertTrue('The dim size of Ascend net output: [2, 2] '
                'is not equal to FX net output: [1, 2, 3]' in context.exception)

    def test_check_cann_aclnn_avaliable(self):
        initialize_graph_engine()
        from torchair.core import _torchair
        check_has_v2 = _torchair.CheckAclnnAvaliable("aclnnTest")
        
    def test_data_dump_generation(self):
        import re

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = Model()
        test_config = torchair.CompilerConfig()
        test_config.debug.data_dump.type = "npy"
        test_config.debug.data_dump.path = "./test_data_dump_generation/dir"
        if os.path.exists(test_config.debug.data_dump.path):
            shutil.rmtree(test_config.debug.data_dump.path)
        test_config.ge_config.export_compile_stat = "0"
        test_npu_backend = torchair.get_npu_backend(compiler_config=test_config)
        test_model = torch.compile(model, backend=test_npu_backend)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        test_model(x, y)
        path = os.path.realpath(f'{test_config.debug.data_dump.path}/data_dump/1')
        self.assertTrue(os.path.isdir(path), f"directory {path} does not exist.")
        file_path = os.path.join(path, os.listdir(path)[0])
        test_files = [f for f in os.listdir(file_path) if f.endswith(".npy")]
        self.assertEqual(len(test_files), 3, f"found data_dump files in {file_path}")

    def test_data_dump_failpath(self):
        import re

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)

        model = Model()
        test_config = torchair.CompilerConfig()
        os.mkdir("./test_data_dump_failpath")
        with open("./test_data_dump_failpath/fail.txt", "w") as f:
            f.write("data dump test")
        test_config.debug.data_dump.type = "npy"
        test_config.debug.data_dump.path = "./test_data_dump_failpath/fail.txt"
        test_config.ge_config.export_compile_stat = "0"
        test_npu_backend = torchair.get_npu_backend(compiler_config=test_config)
        test_model = torch.compile(model, backend=test_npu_backend)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        with self.assertRaises(NotADirectoryError):
            test_model(x, y)

    def test_reset_resource(self):
        class Model(torch.nn.Module):
            def forward(self, input):
                return torch.add(input, 1.0)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x)

        def custom_del(self):
            print("start to release graph")
        GeConcreteGraph.__del__ = custom_del

        with capture_stdout() as stdout:
            torch._dynamo.reset()
        del GeConcreteGraph.__del__

        captured_output = stdout.getvalue()
        self.assertTrue("start to release graph" in captured_output)

if __name__ == '__main__':
    unittest.main()
