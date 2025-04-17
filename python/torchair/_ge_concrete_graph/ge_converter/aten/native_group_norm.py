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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote, specific_op_input_layout, \
    specific_op_output_layout


@declare_supported([
    Support(F32(2, 320, 64, 64), None, None, 2, 320, 4096, 32, 1e-05),
    Support(F16(2, 320, 64, 64), None, None, 2, 320, 4096, 32, 1e-05),
    Support(F32(2, 320, 64, 64), F32(320), F32(320), 2, 320, 4096, 32, 1e-05),
])
@register_fx_node_ge_converter(torch.ops.aten.native_group_norm.default)
def conveter_aten_native_group_norm_default(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    N: Union[int, Tensor],
    C: Union[int, Tensor],
    HxW: Union[int, Tensor],
    group: int,
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)"""
    if weight is None:
        one_value = dtype_promote(1, target_dtype=meta_outputs[0].dtype)
        weight = ge.Fill([C], one_value)
    if bias is None:
        zero_value = dtype_promote(0, target_dtype=meta_outputs[0].dtype)
        bias = ge.Fill([C], zero_value)
    y, mean, variance = ge.GroupNorm(input, weight, bias, num_groups=group, eps=eps, is_training=True)
    specific_op_input_layout(y, indices=[0, 1, 2], layout="NCHW")
    specific_op_output_layout(y, indices=[0, 1, 2], layout="NCHW")
    eps = dtype_promote(eps, target_dtype=meta_outputs[0].dtype)
    rstd = ge.Rsqrt(ge.Add(variance, eps))
    return y, mean, rstd


@register_fx_node_ge_converter(torch.ops.aten.native_group_norm.out)
def conveter_aten_native_group_norm_out(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    N: Union[int, Tensor],
    C: Union[int, Tensor],
    HxW: Union[int, Tensor],
    group: int,
    eps: float,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_group_norm.out(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise NotImplementedError("torch.ops.aten.native_group_norm.out ge_converter is not implemented!")
