from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(4, 5), 0, None, None, 1),
        Support(F32(4, 5), 1, None, None, 1),
        Support(I32(4, 5), 0, None, None, 1),
        Support(I32(4, 5), 1, None, None, 1),
        Support(F32(4, 5), 0, None, 2, 1),
        Support(I32(4, 5), 1, 1, None, 1),
        Support(F32(4, 5), 0, 1, 4, 1),
        Support(I32(4, 5), 1, 1, 4, 2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.slice.Tensor)
def conveter_aten_slice_Tensor(
    self: Tensor,
    dim: int = 0,
    start: Optional[Union[int, Tensor]] = None,
    end: Optional[Union[int, Tensor]] = None,
    step: Union[int, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)"""

    # performance optimization: if the input and output symbolic shape is equal, do not slice
    if hasattr(self, "_symsize") and meta_outputs is not None and hasattr(meta_outputs, "_symsize"):
        if str(self._symsize) == str(meta_outputs._symsize):
            return self

    if start is None:
        start = 0
    if end is None:
        end = 2147483647
    if end is not None and not isinstance(end, Tensor):
        if end == 9223372036854775807:
            end = 2147483647
        elif end > 2147483647:
            raise RuntimeError("ge.StridedSliceV2 does not support shapes exceeding the INT32_MAX!")
    
    dim, start, end, step = dtype_promote(dim, start, end, step, target_dtype=DataType.DT_INT64)
    return ge.StridedSliceV2(self, start, end, axes=dim, strides=step)


@register_fx_node_ge_converter(torch.ops.aten.slice.str)
def conveter_aten_slice_str(
    string: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str"""
    raise RuntimeError("torch.ops.aten.slice.str ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
