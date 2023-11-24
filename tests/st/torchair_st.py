import math
from torchair.core.utils import logger
import logging
from torchair.core.backend import TorchNpuGraph
from torchair.ge_concrete_graph.ge_graph import GeGraph
from torchair.ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import DataType
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.backend import initialize_graph_engine
import torchair
import torch
import unittest
import time
import os
os.environ['TNG_LOG_LEVEL'] = '0'


logger.setLevel(logging.DEBUG)

config = CompilerConfig()
config.aoe_config.aoe_mode = "1"
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

    def test_auto_tune(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.add(x, y)
        model = torch.compile(Model(), backend=npu_backend, dynamic=False)
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

        exec(src)

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
            executor.load(graph.SerializeToString())
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
            executor.load(graph.SerializeToString())
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
            executor.load(graph.SerializeToString())
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
                return inp + a

        def check_graph(concrete_graph):
            num_broadcastto = 0
            for node in concrete_graph.graph.op:
                if node.type == 'BroadcastTo':
                    num_broadcastto += 1

            assert num_broadcastto == 0, f"check number of num_broadcastto{num_broadcastto} == 0 failed"

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


if __name__ == '__main__':
    unittest.main()
