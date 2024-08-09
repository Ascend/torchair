from typing import Any, Dict, List, Tuple, Union, Callable, Optional

from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge._ge_graph import auto_convert_to_tensor
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
from torchair.ge._ge_graph import _ge_dtype_to_ge_proto_dtype

from torchair.ge._ge_graph import Cast, Const, Data
from torchair._ge_concrete_graph.auto_generated_ge_raw_ops import *
from torchair._ge_concrete_graph.dynamic_output_ops import *


@auto_convert_to_tensor([], [])
def NetOutput(inputs: List[Tensor], name=None, *, dependencies=()) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "NetOutput"
    op.name = next_unique_name(name, "NetOutput")

    for dependency in dependencies:
        op.input.append(dependency.controller)

    for i, input in enumerate(inputs):
        op.input.append(input.tensor)
        op.input_desc.append(input.desc)
        op.input_desc[-1].name = f"input{i}"
    return
