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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


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
    raise NotImplementedError("torch.ops.aten.randn.default ge_converter is not implemented!")


@declare_supported(
    [
        Support([3, 4, 5], generator=None),
        Support([3, 4], generator=None, dtype=torch.float16)
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
    if generator is not None:
        raise NotImplementedError("parameter generator should be handled in eager!")
    seed, offset = get_ge_rng_state(philox_num=10)
    key = ge.Unsqueeze(ge.Cast(seed, dst_type=DataType.DT_UINT64), axes=[0])
    offset0 = ge.Const(0, dtype=DataType.DT_INT64)
    offset_list = ge.ConcatV2([offset0, offset], 0, N=2)
    counter = ge.Cast(offset_list, dst_type=DataType.DT_UINT64)
    alg = ge.Cast(1, dst_type=DataType.DT_INT32)
    return ge.StatelessRandomNormalV2(size, key, counter, alg, dtype=meta_outputs.dtype)


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
    raise NotImplementedError("torch.ops.aten.randn.names ge_converter is not implemented!")


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
    raise NotImplementedError("torch.ops.aten.randn.generator_with_names ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randn.out)
def conveter_aten_randn_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randn.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randn.generator_out)
def conveter_aten_randn_generator_out(
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randn.generator_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randn.names_out)
def conveter_aten_randn_names_out(
    size: Union[List[int], Tensor],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randn.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randn.names_out ge_converter is not implemented!")


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
    raise NotImplementedError("torch.ops.aten.randn.generator_with_names_out ge_converter is not implemented!")
