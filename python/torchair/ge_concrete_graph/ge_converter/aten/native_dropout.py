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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.native_dropout.default)
def conveter_aten_native_dropout_default(
    input: Tensor, p: float, train: Optional[bool], meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.native_dropout.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.native_dropout.out)
def conveter_aten_native_dropout_out(
    input: Tensor,
    p: float,
    train: Optional[bool],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    meta_outputs: Union[TensorSpec, List[TensorSpec]] = None
):
    """NB: aten::native_dropout.out(Tensor input, float p, bool? train, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"""
    raise NotImplementedError("torch.ops.aten.native_dropout.out ge_converter is not implemented!")
