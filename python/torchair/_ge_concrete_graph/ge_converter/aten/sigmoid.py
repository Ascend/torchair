from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
 

@declare_supported(
    [
        Support(F32(2, 6, 1, 1)),
        Support(F32(96, 65)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.sigmoid.default)
def conveter_aten_sigmoid_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sigmoid(Tensor self) -> Tensor"""
    return ge.Sigmoid(self)


@register_fx_node_ge_converter(torch.ops.aten.sigmoid.out)
def conveter_aten_sigmoid_out(
    self: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.sigmoid.out is redundant before pytorch 2.1.0,"
                       "might be supported in future version.")
