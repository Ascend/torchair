from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import F32, F16, BF16, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024)),
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=-1),
        Support(F32(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=0.1),
        Support(F16(1024, 1024), F16(1024, 1024), F16(1024, 1024), value=0.1),
        Support(F32(1024, 1024), F16(1024, 1024), F16(1024, 1024), value=0.1),
        Support(F16(1024, 1024), F32(1024, 1024), F32(1024, 1024), value=0.1),
        Support(BF16(1024, 1024), BF16(1024, 1024), BF16(1024, 1024), value=0.1),
        Support(F32(1024), F32(1024), F32(1024))
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.addcdiv.default)
def conveter_aten_addcdiv_default(
    self: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor"""
    self, tensor1, tensor2, value = dtype_promote(self, tensor1, tensor2, value, target_dtype=meta_outputs.dtype)
    return ge.Addcdiv(self, tensor1, tensor2, value)


@register_fx_node_ge_converter(torch.ops.aten.addcdiv.out)
def conveter_aten_addcdiv_out(
    self: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addcdiv.out ge_converter is not implemented!")
