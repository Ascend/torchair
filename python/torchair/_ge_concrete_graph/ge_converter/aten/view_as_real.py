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
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, C64, \
    BOOL, Support


@declare_supported(
    [
        Support(C64(2, 2)),
        Support(C64(2, 8, 16)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.view_as_real.default)
def conveter_aten_view_as_real_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::view_as_real(Tensor(a) self) -> Tensor(a)"""
    real = ge.Real(self)
    imag = ge.Imag(self)
    result = ge.Pack([real, imag], N=2, axis=-1)
    return result
