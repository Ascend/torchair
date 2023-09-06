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
        x = torch.randn(512, 1024, 1024)
        y = torch.randn(512, 1024, 1024)
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
                return x

        model = Model()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        x = torch.randn(2, 2)
        model(x, 2, 3)
        model(x, 3, 4)

    def test_ge_api_support_position_passin_by_kv(self):
        # shape is position input of ge.Empty, check not raise when pass shape by k-v
        ge.Empty(shape=ge.Const(1))

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
            result = executor.run((x, y))


if __name__ == '__main__':
    unittest.main()
