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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, torch_type_to_ge_type, declare_supported, \
    torch_type_to_ge_type
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote



@declare_supported([
    Support(BOOL(2, 2)),
    Support(I32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.bitwise_not.default)
def conveter_aten_bitwise_not_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::bitwise_not(Tensor self) -> Tensor"""
    if self.dtype == torch_type_to_ge_type(torch.bool):
        output = ge.LogicalNot(self)
    else:
        output = ge.Invert(self)
    return output


@register_fx_node_ge_converter(torch.ops.aten.bitwise_not.out)
def conveter_aten_bitwise_not_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bitwise_not.out ge_converter is not implemented!")
