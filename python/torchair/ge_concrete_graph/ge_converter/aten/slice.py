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

import torch
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


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
    if end is not None and not isinstance(end, Tensor):
        if end == 9223372036854775807:
            end = 2147483647
        elif end > 2147483647:
            raise NotImplementedError("ge.StridedSlice does not support shapes exceeding the INT32_MAX!")

    mask = [False for _ in range(self.rank)]
    mask[dim] = True
    mask = ge.Const(mask, DataType.DT_BOOL)

    ge_begin = [0 for _ in range(self.rank)]
    ge_end = [2147483647 for _ in range(self.rank)]
    ge_strides = [1 for _ in range(self.rank)]

    if start is not None:
        if isinstance(start, Tensor):
            ge_begin, start = dtype_promote(ge_begin, start, target_dtype=DataType.DT_INT64)
            ge_begin = ge.MaskedFill(ge_begin, mask, start)
        else:
            ge_begin[dim] = start
    if end is not None:
        if isinstance(end, Tensor):
            ge_end, start = dtype_promote(ge_end, end, target_dtype=DataType.DT_INT64)
            ge_end = ge.MaskedFill(ge_end, mask, end)
        else:
            ge_end[dim] = end
    if isinstance(step, Tensor):
        ge_strides, step = dtype_promote(ge_strides, step, target_dtype=DataType.DT_INT64)
        ge_strides = ge.MaskedFill(ge_strides, mask, step)
    else:
        ge_strides[dim] = step

    return ge.StridedSlice(self, ge_begin, ge_end, ge_strides)


@register_fx_node_ge_converter(torch.ops.aten.slice.str)
def conveter_aten_slice_str(
    string: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str"""
    raise NotImplementedError("torch.ops.aten.slice.str ge_converter is not implemented!")
