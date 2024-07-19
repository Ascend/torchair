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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported, \
    register_checkpoint_func
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, get_ge_rng_state
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support


@register_checkpoint_func(torch.ops.aten.multinomial.default)
def multinomial_default_checkpoint(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
    rng_state: Optional[Tensor] = None
):
    cur_dim = self.rank
    if cur_dim not in {1, 2}:
        raise RuntimeError("torch.ops.aten.multinomial.default, dim of input tensor can only be 1 or 2.")
    if rng_state is None:
        seed, offset = get_ge_rng_state(philox_num=10, gen=generator)
    else:
        seed, offset = ge.Unpack(rng_state, num=2, axis=0)
    return (seed, offset), ge.MultinomialWithReplacement(self, seed, offset, numsamples=num_samples)


@declare_supported([
    Support(F16(1, 64), num_samples=1, replacement=True)
])
@register_fx_node_ge_converter(torch.ops.aten.multinomial.default)
def conveter_aten_multinomial_default(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor"""
    _, result = multinomial_default_checkpoint(self, num_samples, replacement, generator, meta_outputs)
    return result


@register_fx_node_ge_converter(torch.ops.aten.multinomial.out)
def conveter_aten_multinomial_out(
    self: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.multinomial.out ge_converter is not implemented!")
