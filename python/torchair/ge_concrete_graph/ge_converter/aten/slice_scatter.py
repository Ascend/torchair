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
    Support(F32(22), F32(10), 0, 0, 10)
])
@register_fx_node_ge_converter(torch.ops.aten.slice_scatter.default)
def conveter_aten_slice_scatter_default(
    self: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor"""
    if start is None:
        start = 0
    if end is None:
        end = 9223372036854775807
    if (isinstance(start, int) and start == 0) and (
            isinstance(end, int) and end == 9223372036854775807) and (
            isinstance(step, int) and step == 1):
        return ge.Identity(src)

    input_sizes = ge.Shape(self)
    if isinstance(end, int) and end == sys.maxsize:
        end = ge.Gather(input_sizes, dim)

    input_sizes, start, limit, delta = dtype_promote(input_sizes, start, end, \
                                                     step, target_dtype=torch_type_to_ge_type(torch.int32))
    
    dims_to_expand = [i for i in range(src.rank)]
    dims_to_expand.remove(dim)
    idx = ge.Range(start, limit, delta)

    if dims_to_expand:
        idx_unsqueezed = ge.Unsqueeze(idx, axes=dims_to_expand)
        idx_expanded = ge.Expand(idx_unsqueezed, ge.Shape(src))
    else:
        idx_expanded = idx

    return ge.ScatterElements(self, idx_expanded, src, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.slice_scatter.out)
def conveter_aten_slice_scatter_out(
    self: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::slice_scatter.out(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.slice_scatter.out ge_converter is not implemented!")
