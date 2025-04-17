from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported, \
    register_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state, torch_type_to_ge_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


def get_dtype(dtype: Optional[int] = None, meta_outputs: TensorSpec = None):
    if dtype is not None:
        return torch_type_to_ge_type(dtype)
    if isinstance(meta_outputs, (list, tuple)):
        return meta_outputs[-1].dtype
    return meta_outputs.dtype


@register_checkpoint_func([torch.ops.aten.randn.default, torch.ops.aten.randn.generator])
def randn_checkpoint(
    size: Union[List[int], Tensor],
    generator: Optional[Generator] = None,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    if rng_state is None:
        seed, offset = get_ge_rng_state(philox_num=10, gen=generator)
    else:
        seed, offset = ge.Unpack(rng_state, num=2, axis=0)
    key = ge.Unsqueeze(ge.Cast(seed, dst_type=DataType.DT_UINT64), axes=[0])
    offset0 = ge.Const(0, dtype=DataType.DT_INT64)
    offset_list = ge.ConcatV2([offset0, offset], 0, N=2)
    counter = ge.Cast(offset_list, dst_type=DataType.DT_UINT64)
    alg = ge.Cast(1, dst_type=DataType.DT_INT32)
    out_dtype = get_dtype(dtype, meta_outputs)
    return (seed, offset), ge.StatelessRandomNormalV2(size, key, counter, alg, dtype=out_dtype)


@declare_supported(
    [
        Support([3, 4, 5], device='npu'),
        Support([3, 4], dtype=torch.float16, device='npu')
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.randn.default)
def conveter_aten_randn_default(
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    _, result = randn_checkpoint(size, None, dtype, layout, device, pin_memory, meta_outputs, None)
    return result


@declare_supported(
    [
        Support([3, 4, 5], generator=None, device='npu'),
        Support([3, 4], generator=None, dtype=torch.float16, device='npu')
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.randn.generator)
def conveter_aten_randn_generator(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    _, result = randn_checkpoint(size, generator, dtype, layout, device, pin_memory, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.randn.names)
def conveter_aten_randn_names(
    size: Union[List[int], Tensor],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.randn.names ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randn.generator_with_names)
def conveter_aten_randn_generator_with_names(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.randn.generator_with_names ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randn.out)
def conveter_aten_randn_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randn.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randn.generator_out)
def conveter_aten_randn_generator_out(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randn.generator_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randn.names_out)
def conveter_aten_randn_names_out(
    size: Union[List[int], Tensor],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randn.names_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randn.generator_with_names_out)
def conveter_aten_randn_generator_with_names_out(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randn.generator_with_names_out ge_converter is not supported!")
