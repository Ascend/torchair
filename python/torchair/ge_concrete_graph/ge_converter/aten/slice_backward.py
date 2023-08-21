from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import sys

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import F32, F16, Support

@declare_supported([
    Support(F32(3, 4), [3, 8], 1, 0, 8, 2),
    Support(F16(3, 4), [3, 8], 1, 0, 8, 2),
])
@register_fx_node_ge_converter(torch.ops.aten.slice_backward.default)
def conveter_aten_slice_backward_default(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    start: Union[int, Tensor],
    end: Union[int, Tensor],
    step: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor"""

    if isinstance(input_sizes, list) and isinstance(end, int) and end == sys.maxsize:
        end = input_sizes[dim]

    input_sizes, start, limit, delta = dtype_promote(input_sizes, start, end, step, target_dtype=torch_type_to_ge_type(torch.int32))
    
    grad_input = ge.Fill(input_sizes, ge.Const(0., dtype=grad_output.dtype))
    dims_to_expand = [i for i in range(grad_output.rank)]
    dims_to_expand.remove(dim)
    idx = ge.Range(start, limit, delta)

    if dims_to_expand:
        idx_unsqueezed = ge.Unsqueeze(idx, axes=dims_to_expand)
        idx_expanded = ge.Expand(idx_unsqueezed, ge.Shape(grad_output))
    else:
        idx_expanded = idx

    return ge.ScatterElements(grad_input, idx_expanded, grad_output, axis=dim)    


@register_fx_node_ge_converter(torch.ops.aten.slice_backward.out)
def conveter_aten_slice_backward_out(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    start: Union[int, Tensor],
    end: Union[int, Tensor],
    step: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::slice_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.slice_backward.out ge_converter is not implemented!")
