from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
from torch.library import Library, impl
import torch_npu
import torchair
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes, compat_as_bytes_list


m = Library("npu_define", "DEF")
m.define("custom_op(Tensor input1, Tensor input2) -> Tensor")


@impl(m, "custom_op", "PrivateUse1")
def plug_custom_op(
        x: torch.Tensor,
        y: torch.Tensor,
):
    return x + y


@impl(m, "custom_op", "Meta")
def custom_op_meta(x, y):
    return torch.empty_like(x)


def Add(x1: Tensor, x2: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(Add)\n
  .INPUT(x1, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
  .INPUT(x2, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
  .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_STRING}))\n
  """

    op = get_default_ge_graph().op.add()
    op.type = "Add"
    op.name = next_unique_name(node_name, "Add")

    # process dependices
    for dependency in dependencies:
        op.input.append(dependency.controller)

    # process inputs
    op.input.append(x1.tensor)
    op.input_desc.add().CopyFrom(x1.desc)
    op.input_desc[-1].name = "x1"
    op.input.append(x2.tensor)
    op.input_desc.add().CopyFrom(x2.desc)
    op.input_desc[-1].name = "x2"

    # process attrs

    # process outputs
    output_index = 0
    op.output_desc.add().name = "y"
    y = Tensor(op, output_index)
    output_index += 1

    # return outputs
    return y


@register_fx_node_ge_converter(torch.ops.npu_define.custom_op.default)
def conveter_custom_op(
        input1: Tensor,
        input2: Tensor,
        out: Tensor = None,
        meta_outputs: Any = None):
    input1, input2 = dtype_promote(input1, input2, target_dtype=meta_outputs.dtype)
    return Add(input1, input2)
