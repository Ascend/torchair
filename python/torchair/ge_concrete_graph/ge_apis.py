from typing import Any, Dict, List, Tuple, Union, Callable, Optional

from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge_concrete_graph.ge_graph import auto_convert_to_tensor
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, compat_as_bytes_list
from torchair.ge_concrete_graph.ge_graph import trans_to_list_list_int, trans_to_list_list_float

from torchair.ge_concrete_graph.ge_graph import Const
from torchair.ge_concrete_graph.auto_generated_ge_raw_ops import *
from torchair.ge_concrete_graph.dynamic_output_ops import *


def Data(*, index: int, dtype: int, shape: List[int] = None, format: str = "ND", name: str = None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Data"
    op.name = next_unique_name(name, "Data")
    op.attr["index"].i = index

    desc = op.output_desc.add()
    desc.name = "y"
    desc.dtype = dtype
    desc.layout = format
    if shape is not None:
        desc.shape.dim.extend(shape)
    else:
        desc.shape.dim.extend([-2])

    op.input_desc.add().CopyFrom(desc)
    op.input_desc[-1].name = "x"

    return Tensor(op)


def NetOutput(inputs: List[Tensor], name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "NetOutput"
    op.name = next_unique_name(name, "NetOutput")
    for i, input in enumerate(inputs):
        op.input.append(input.tensor)
        op.input_desc.append(input.desc)
        op.input_desc[-1].name = f"input{i}"
    return
