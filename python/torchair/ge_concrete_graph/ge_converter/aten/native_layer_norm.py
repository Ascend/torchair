import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm.default)
def conveter_aten_native_layer_norm_default(
        input: Tensor,
        normalized_shape: Union[List[int], Tensor],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.native_layer_norm.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm.out)
def conveter_aten_native_layer_norm_out(
        input: Tensor,
        normalized_shape: Union[List[int], Tensor],
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::native_layer_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten.native_layer_norm.out ge converter is not implement!")


