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


@register_checkpoint_func([torch.ops.aten.randint.default])
def randint_checkpoint(
        high: Union[int, Tensor],
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

    ge_result = ge.StatelessRandomUniformV2(size, key, counter, alg, dtype=DataType.DT_FLOAT)
    mul0 = ge.Mul(ge_result, high)
    if dtype == DataType.DT_UINT8:
        return (seed, offset), ge.Cast(mul0, dst_type=DataType.DT_INT64)

    dtype = torch_type_to_ge_type(dtype)
    if dtype == DataType.DT_BOOL:
        return (seed, offset), ge.Fill(ge.Shape(ge_result), ge.Cast(0 if high == 0 else 1, dst_type=dtype))
    return (seed, offset), ge.Cast(ge.Cast(mul0, dst_type=DataType.DT_INT64), dst_type=dtype)


@declare_supported(
    [
        Support(5, [30, 40, 50], device='npu'),
        Support(4, [30, 40], device='npu'),
        Support(3, [30], device='npu')
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.randint.default)
def conveter_aten_randint_default(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if isinstance(size, Tensor):
        raise RuntimeError("torch.ops.aten.randint.default ge converter is not supported when size is tensor.")
    if not isinstance(high, Tensor) and high <= 0:
        raise RuntimeError(f"from must be less than high, but now from is 0, high is {high}")

    _, result = randint_checkpoint(high, size, None, dtype, layout, device, pin_memory, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.randint.generator)
def conveter_aten_randint_generator(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.randint.generator ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low)
def conveter_aten_randint_low(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.randint.low ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_generator)
def conveter_aten_randint_low_generator(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.randint.low_generator ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.out)
def conveter_aten_randint_out(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randint.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.generator_out)
def conveter_aten_randint_generator_out(
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randint.generator_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_out)
def conveter_aten_randint_low_out(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randint.low_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.randint.low_generator_out)
def conveter_aten_randint_low_generator_out(
    low: Union[int, Tensor],
    high: Union[int, Tensor],
    size: Union[List[int], Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.randint.low_generator_out ge_converter is not supported!")
