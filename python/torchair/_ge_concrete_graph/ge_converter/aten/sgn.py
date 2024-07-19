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
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.sgn.default)
def conveter_aten_sgn_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sgn(Tensor self) -> Tensor"""
    # The complex data type is not supported, 28 represent the unknown dtype!
    if self.dtype == 28:
        raise NotImplementedError("torch.ops.aten.sgn.tensor ge_converter with input of complex is not implemented!")
    return ge.Sign(self)


@register_fx_node_ge_converter(torch.ops.aten.sgn.out)
def conveter_aten_sgn_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sgn.out ge_converter is not supported!")
