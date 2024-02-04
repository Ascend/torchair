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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(2, 2), [I64(1, 2), I64(1, 2)], F32(1, 2)),
])
@register_fx_node_ge_converter(torch.ops.aten.index_put.default)
def conveter_aten_index_put_default(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    values = dtype_promote(values, target_dtype=self.dtype)
    mask = [1 if indice else 0 for indice in indices]
    indice = [i for i in indices if i]
    self = ge.TensorMove(self)
    return ge.IndexPutImpl(self, values, indice, indices_mask=mask, accumulate=accumulate)


@register_fx_node_ge_converter(torch.ops.aten.index_put.out)
def conveter_aten_index_put_out(
    self: Tensor,
    indices: List[Optional[Tensor]],
    values: Tensor,
    accumulate: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.index_put.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.index_put.hacked_twin)
def conveter_aten_index_put_hacked_twin(
    self: Tensor,
    indices: List[Tensor],
    values: Tensor,
    accumulate: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::index_put.hacked_twin(Tensor self, Tensor[] indices, Tensor values, bool accumulate=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.index_put.hacked_twin ge_converter is not implemented!")
