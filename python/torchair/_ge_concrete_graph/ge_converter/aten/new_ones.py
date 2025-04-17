from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), (2, 3)),
    Support(F32(2, 2), (2, 3), dtype=torch.int),
    Support(F16(2, 2), (2, 3)),
    Support(I32(2, 2), (2, 3)),
])
@register_fx_node_ge_converter(torch.ops.aten.new_ones.default)
def conveter_aten_new_ones_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = self.dtype
    else:
        dtype = torch_type_to_ge_type(dtype)

    if layout is not None and layout != torch.strided:
        raise NotImplementedError("torch.ops.aten.new_ones.default ge_converter is only supported on dense  tensor!")

    return ge.Fill(size, ge.Cast(1., dst_type=dtype))


@register_fx_node_ge_converter(torch.ops.aten.new_ones.out)
def conveter_aten_new_ones_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_ones.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.new_ones.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
