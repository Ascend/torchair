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
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.split.Tensor)
def conveter_aten_split_Tensor(
    self: Tensor, split_size: Union[int, Tensor], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]"""
    split_sizes = split_size
    if dim > 2147483647:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT64)
    else:
        dim = dtype_promote(dim, target_dtype=DataType.DT_INT32)
    if isinstance(split_sizes, int):
        split_sizes = [split_size for _ in range(len(meta_outputs))]
        split_sizes[-1] = -1
        split_sizes = dtype_promote(split_sizes, target_dtype=DataType.DT_INT64)
        return ge.SplitV(self, split_sizes, dim, num_split=len(meta_outputs))
    elif isinstance(split_sizes, Tensor):
        tensors = [split_size for _ in range(len(meta_outputs) - 1)]
        split_sizes = ge.ConcatV2(tensors, concat_dim=0, N=len(meta_outputs))
        split_sizes = ge.ConcatV2([split_sizes, -1], concat_dim=0, N=2)
        split_sizes = dtype_promote(split_sizes, target_dtype=DataType.DT_INT64)
        return ge.SplitV(self, split_sizes, dim, num_split=len(meta_outputs))


@register_fx_node_ge_converter(torch.ops.aten.split.sizes)
def conveter_aten_split_sizes(
    self: Tensor,
    split_size: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.sizes ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.str)
def conveter_aten_split_str(
    self: str, separator: Optional[str] = None, max: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::split.str(str self, str? separator=None, int max=-1) -> str[]"""
    raise NotImplementedError("torch.ops.aten.split.str ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.split.default)
def conveter_aten_split_default(
    self: Tensor, split_sizes: List[int], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::split(Tensor(a -> *) self, int[] split_sizes, int dim=0) -> Tensor(a)[]"""
    raise NotImplementedError("torch.ops.aten.split.default ge_converter is not implemented!")
