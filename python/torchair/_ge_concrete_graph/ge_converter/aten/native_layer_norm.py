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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F16(2, 1024), [1024], F16(1024), F16(1024), 1e-5),
        Support(F16(2, 1024), [1024], None, None, 1e-5),
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
    if weight is None:
        weight = ge.Fill(ge.Const(normalized_shape, dtype=DataType.DT_INT32), 
                         ge.Cast(1., dst_type=input.dtype))
        
    if bias is None:
        bias = ge.Fill(ge.Const(normalized_shape, dtype=DataType.DT_INT32), 
                       ge.Cast(0., dst_type=input.dtype))
    
    return ge.LayerNormV4(input, normalized_shape, gamma=weight, beta=bias, epsilon=eps)


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
    raise RuntimeError("torch.ops.aten.native_layer_norm.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
