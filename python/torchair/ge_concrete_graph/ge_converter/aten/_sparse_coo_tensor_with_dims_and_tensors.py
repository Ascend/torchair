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
from torch import Generator, contiguous_format, inf, strided
from torch.types import Device, Number, SymInt, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.default)
def conveter_aten__sparse_coo_tensor_with_dims_and_tensors_default(
    sparse_dim: int,
    dense_dim: int,
    size: Union[List[int], Tensor],
    indices: Tensor,
    values: Tensor,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = False,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor"""
    raise NotImplementedError(
        "torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.default ge_converter is not implemented!"
    )


@register_fx_node_ge_converter(torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.out)
def conveter_aten__sparse_coo_tensor_with_dims_and_tensors_out(
    sparse_dim: int,
    dense_dim: int,
    size: Union[List[int], Tensor],
    indices: Tensor,
    values: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_sparse_coo_tensor_with_dims_and_tensors.out(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError(
        "torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors.out ge_converter is not implemented!"
    )
