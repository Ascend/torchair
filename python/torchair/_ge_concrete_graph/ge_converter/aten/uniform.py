import math
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


@register_checkpoint_func([torch.ops.aten.uniform.default])
def _uniform_checkpoint(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    size = ge.Shape(self)
    if rng_state is None:
        seed, offset = get_ge_rng_state(philox_num=10, gen=generator)
    else:
        seed, offset = ge.Unpack(rng_state, num=2, axis=0)
    key = ge.Unsqueeze(ge.Cast(seed, dst_type=DataType.DT_UINT64), axes=[0])
    offset0 = ge.Const(0, dtype=DataType.DT_INT64)
    offset_list = ge.ConcatV2([offset0, offset], 0, N=2)
    counter = ge.Cast(offset_list, dst_type=DataType.DT_UINT64)
    alg = ge.Cast(1, dst_type=DataType.DT_INT32)
    result = ge.StatelessRandomUniformV2(size, key, counter, alg, dtype=self.dtype)
    if math.isclose(from_, 0.0, rel_tol=1e-9) and math.isclose(to, 1.0, rel_tol=1e-9):
        return (seed, offset), result
    # U(0~1) --> U(from_~to)
    from_, to = dtype_promote(from_, to, target_dtype=self.dtype)
    tmp = ge.Sub(ge.Mul(result, from_), from_)
    result = ge.Sub(ge.Mul(result, to), tmp)
    return (seed, offset), result


@declare_supported(
    [
        Support(F32(2, 2)),
        Support(F16(1, 2), 1., 2.),
        Support(F32(20,)),
        Support(F32(20,), 0.5, 1.),
        Support(F16(3, 2), 0, 2.),
        Support(F16(3, 2), 0.5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.uniform.default)
def conveter_aten_uniform_default(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::uniform(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> Tensor"""
    _, result = _uniform_checkpoint(self, from_, to, generator, meta_outputs, None)
    return result


@register_fx_node_ge_converter(torch.ops.aten.uniform.out)
def conveter_aten_uniform_out(
    self: Tensor,
    from_: float = 0.0,
    to: float = 1.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::uniform.out(Tensor self, float from=0., float to=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.uniform.out ge_converter is not implemented!")
