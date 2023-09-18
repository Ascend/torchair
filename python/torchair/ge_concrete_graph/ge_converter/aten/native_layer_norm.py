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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F16(2, 1024), [1024], F16(1024), F16(1024), 1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm.default)
def conveter_aten_native_layer_norm_default(
    input: Tensor,
    normalized_shape: Union[List[int], Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""
    if weight is None or bias is None or isinstance(normalized_shape, Tensor):
        raise NotImplementedError("torch.ops.aten.native_layer_norm.default ge_converter is not implemented!")

    norm_dim = len(normalized_shape)
    input_dim = input.rank
    begin_axis = input_dim - norm_dim
    return ge.LayerNormV3(input, weight, bias, begin_norm_axis=begin_axis, begin_params_axis=begin_axis, epsilon=eps)


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_layer_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten.native_layer_norm.out ge_converter is not implemented!")
