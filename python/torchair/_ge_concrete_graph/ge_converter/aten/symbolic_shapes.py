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
from math import sqrt
import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote
try:
    from torch import sym_sqrt
except ImportError:
    from torch.fx.experimental.symbolic_shapes import sym_sqrt


@register_fx_node_ge_converter(sym_sqrt)
def conveter_sym_sqrt(
        self: Union[Number, Tensor],
        meta_outputs: TensorSpec = None
):
    if not isinstance(self, Tensor):
        return sqrt(self)
    return ge.Sqrt(self)