import unittest
import os
import logging
import torch
os.environ['TNG_LOG_LEVEL'] = '0'
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair.inference import set_dim_gears
from torchair.core._backend import TorchNpuGraph
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import GeGraph, DataType
from torchair.core._backend import initialize_graph_engine
from torchair.ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph
from torchair.inference._gear_utils import generate_dynamic_dims_option
from torchair.ge_concrete_graph.ge_graph import trans_to_list_list_int

logger.setLevel(logging.DEBUG)


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



class TorchairSt(unittest.TestCase):
    def test_mark_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * 2
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.ones([8, 8])
        set_dim_gears(x, {0: [8, 4]})
        def my_warp_function(named_inputs_info, config):
            result = generate_dynamic_dims_option(named_inputs_info, config)
            self.assertTrue(result == {'ge.inputShape': 'arg2_1:-1,8',
                                       'ge.dynamicDims': '8;4',
                                       'ge.dynamicNodeType': '1'})
            return result
        with unittest.mock.patch('torchair.inference._gear_utils.generate_dynamic_dims_option', my_warp_function):
            model(x)
            model(x)

    def test_product(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * 2
        config = CompilerConfig()
        config.inference_config.dynamic_gears_merge_policy = "product"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.ones([8, 8])
        set_dim_gears(x, {1: [8, 12], 0: [8, 4]})
        def my_warp_function(named_inputs_info, config):
            result = generate_dynamic_dims_option(named_inputs_info, config)
            self.assertTrue(result == {'ge.inputShape': 'arg2_1:-1,-1',
                                        'ge.dynamicDims': '8,8;8,12;4,8;4,12',
                                        'ge.dynamicNodeType': '1'})
            return result
        with unittest.mock.patch('torchair.inference._gear_utils.generate_dynamic_dims_option', my_warp_function):
            model(x)
            model(x)

    def test_muti_gear_npu_executor(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Mul(x, y)
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"ge.dynamicDims": "8;12",
                                          "ge.dynamicNodeType": "1",
                                          "ge.inputShape": "arg:-1,2"})
            executor.set_hint_shape([[-1, 2], []], [[-1, 2]])
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            y = torch.ones([], dtype=torch.int32)
            z = executor.run([x, y], [x])
            z = executor.run([x, y], [x])
            self.assertTrue(z[0] is x)

    def test_muti_gear_npu_executor_muti_output(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            w = ge.Data(index=2, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            z1 = ge.Mul(x, y)
            z2 = ge.Mul(w, y)
            output = ge.NetOutput([z1, z2])

            set_graph_output_dtypes(graph, [DataType.DT_INT32, DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"ge.dynamicDims": "2;4",
                                          "ge.dynamicNodeType": "1",
                                          "ge.inputShape": "arg:-1,2;;2,2"})
            executor.set_hint_shape([[-1, 2], [], [2, 2]], [[-1, 2], [2, 2]])
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            y = torch.ones([], dtype=torch.int32)
            w = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            z1, z2 = executor.run([x, y, w])
            z1, z2 = executor.run([x, y, w])


if __name__ == '__main__':
    unittest.main()
