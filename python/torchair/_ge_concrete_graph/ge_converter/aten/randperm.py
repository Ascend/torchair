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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, register_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state


@register_checkpoint_func([torch.ops.aten.randperm.default])
def randperm_checkpoint(
        n: Union[int, Tensor],
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
    layout = 1
    if isinstance(meta_outputs, (list, tuple)):
        out_dtype = meta_outputs[-1].dtype
    else:
        out_dtype = meta_outputs.dtype
    ge_result = ge.StatelessRandperm(n, seed, offset, layout=layout, dtype=out_dtype)
    return (seed, offset), ge_result


@register_fx_node_ge_converter(torch.ops.aten.randperm.default)
def conveter_aten_randperm_default(
    n: Union[int, Tensor],
    *,
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randperm(SymInt n, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if not isinstance(n, Tensor) and n <= 0:
        raise RuntimeError(f"from must be less than to, but now from is 0, to is {n}")

    _, result = randperm_checkpoint(n, None, dtype, layout, device, pin_memory, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.randperm.generator)
def conveter_aten_randperm_generator(
    n: Union[int, Tensor],
    *,
    generator: Optional[Generator],
    dtype: Optional[int] = 4,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.randperm.generator ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randperm.out)
def conveter_aten_randperm_out(
    n: Union[int, Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randperm.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.randperm.generator_out)
def conveter_aten_randperm_generator_out(
    n: Union[int, Tensor],
    *,
    generator: Optional[Generator],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.randperm.generator_out ge_converter is not implemented!")
