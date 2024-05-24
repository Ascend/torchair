from torchair.core.utils import logger
from torchair.core._backend import initialize_graph_engine
from torchair.ge_concrete_graph.ge_graph import DataType
from torchair.ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.ge_graph import GeGraph
from torchair.core._backend import TorchNpuGraph
import logging
import torch
import torch_npu
torch_npu.npu.set_device(0)


logger.setLevel(logging.DEBUG)

initialize_graph_engine()


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


with GeGraph() as graph:
    x = ge.Data(index=0, shape=[2, 2],
                dtype=DataType.DT_INT32, placement='NPU')
    y = ge.Data(index=1, shape=[], dtype=DataType.DT_INT32, placement='CPU')
    z = ge.Add(x, y)
    output = ge.NetOutput([z])

    set_graph_output_dtypes(graph, [DataType.DT_INT32])

    executor = TorchNpuGraph()
    executor.load(graph.SerializeToString())
    executor.compile()

    x = torch.ones([2, 2], dtype=torch.int32).npu()
    y = torch.ones([], dtype=torch.int32)
    output = executor.run((x, y))

    print(output)
