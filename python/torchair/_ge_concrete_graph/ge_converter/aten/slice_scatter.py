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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair._ge_concrete_graph.supported_declaration import F32, F16, Support
from torchair._ge_concrete_graph.ge_converter.aten.slice_backward import check_if_skip


def static_slice_scatter_step1(
    self: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
):
    if check_if_skip(start, end, step):
        return src
    int_max = 2147483647
    if not isinstance(end, Tensor) and end != sys.maxsize and end > int_max:
        raise NotImplementedError("ge.StridedSliceV2 does not support shapes exceeding the INT32_MAX!")

    dim = dim % self.rank
    limit = self.symsize[dim]
    if end < 0:
        end = end % limit
    else:
        end = min(end, int_max)

    if start < 0:
        start = start % limit
    else:
        start = min(start, int_max)
    
    start = min(start, end)
    to_pad = ge.StridedSliceV2(self, 0, start, axes=dim, strides=step)
    to_pad2 = ge.StridedSliceV2(self, end, int_max, axes=dim, strides=step)
    output = ge.ConcatV2([to_pad, src, to_pad2], concat_dim=dim, N=3)
    return output

@declare_supported([
    Support(F32(22), F32(10), 0, 0, 10),
    Support(F32(10), F32(10), 0, 0, 20),
    Support(F32(10, 10), F32(10, 2), 1, 1, 3),
    Support(F32(10, 10), F32(10, 1), 1, -1, 9223372036854775807),
    Support(F32(10, 10), F32(10, 10), 1, 0, 9223372036854775807),
    Support(F32(10, 10), F32(10, 10)),
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
        end = sys.maxsize

    if not (isinstance(start, Tensor) or isinstance(end, Tensor) or \
        isinstance(step, Tensor)) and step == 1:
        return static_slice_scatter_step1(self, src, dim, start, end, step)
    if (isinstance(start, int) and start == 0) and (
            isinstance(end, int) and end == sys.maxsize) and (
            isinstance(step, int) and step == 1):
        return ge.Identity(src)

    input_sizes = ge.Shape(self)
    if isinstance(end, int) and end == sys.maxsize:
        end = ge.Gather(input_sizes, dim)

    input_sizes, start, limit, delta = dtype_promote(input_sizes, start, end, \
                                                     step, target_dtype=torch_type_to_ge_type(torch.int32))
    
    dims_to_expand = [i for i in range(src.rank)]
    dims_to_expand.remove(dim)

    limit = ge.Minimum(ge.Gather(input_sizes, dim), limit)
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
    raise RuntimeError("torch.ops.aten.slice_scatter.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
