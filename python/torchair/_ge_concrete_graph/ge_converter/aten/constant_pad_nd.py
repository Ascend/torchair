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
        Support(F32(12, 39, 12), [0, 0, 0, 1, 0, 0], 0.0),
        Support(F32(2, 8, 13, 13), [0, -1, 0, -1]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.constant_pad_nd.default)
def conveter_aten_constant_pad_nd_default(
    self: Tensor,
    pad: Union[List[int], Tensor],
    value: Union[Number, Tensor] = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor"""
    if isinstance(pad, Tensor):
        raise NotImplementedError("When pad is Tensor, torch.ops.aten.constant_pad_nd.default ge_converter is not implemented!")
    if len(pad) % 2 != 0:
        raise AssertionError(f"Length of pad must be even but instead it equals {len(pad)}")
    if not (self.rank >= (len(pad) / 2)):
        raise AssertionError("Length of pad should be no more than twice the number of dimensions of the input. ")
    paddings = [0] * (2 * self.rank)
    if len(pad) <= len(paddings):
        paddings[0: len(pad)] = pad
    else:
        paddings = pad[0: len(paddings)]
    pads = []
    paddings_len = len(paddings)
    while paddings_len > 0:
        pads.append(paddings[paddings_len - 2])
        pads.append(paddings[paddings_len - 1])
        paddings_len -= 2
    return ge.PadV3(self, pads, value)


@register_fx_node_ge_converter(torch.ops.aten.constant_pad_nd.out)
def conveter_aten_constant_pad_nd_out(
    self: Tensor,
    pad: Union[List[int], Tensor],
    value: Union[Number, Tensor] = 0,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.constant_pad_nd.out ge_converter is not implemented!")
