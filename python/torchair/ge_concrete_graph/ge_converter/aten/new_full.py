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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@declare_supported(
    [
        Support(F32(6, 4), [6, 4], 30, dtype=torch.float16),
        Support(F32(6, 4), [6, 4], 30),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.new_full.default)
def conveter_aten_new_full_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is not None:
        fill_value = dtype_promote(fill_value, target_dtype=dtype)
    else:
        fill_value = dtype_promote(fill_value, target_dtype=self.dtype)
    if layout is not None and layout != torch.strided:
        raise NotImplementedError("torch.ops.aten.new_full.default ge_converter is only supported on dense tensor now!")
    return ge.Fill(size, fill_value)


@register_fx_node_ge_converter(torch.ops.aten.new_full.out)
def conveter_aten_new_full_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_full.out(Tensor self, SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.new_full.out ge_converter is not implemented!")
