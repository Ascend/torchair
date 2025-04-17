from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch.types import Number
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter, declare_supported
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import F32, F16, I64, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.core.utils import logger


@declare_supported([
    # There is precision error with test case: Support(F32(8, 16, 512, 1023), 3, I64(512), F32(8, 16, 512, 512)), 
    # This problem is being resolved
    Support(F32(8, 16, 512, 1023), 3, I64(512), F32(8, 16, 512, 512)),
    Support(F16(5, 3), 0, I64(3), F16(3, 3)),
    Support(F32(5, 3), 0, I64(3), F32(3, 3))
])
@register_fx_node_ge_converter(torch.ops.aten.index_add.default)
def conveter_aten_index_add_default(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor"""
    # The dtype of alpha and self should be the same, otherwise it will cause Segmentation fault
    alpha = dtype_promote(alpha, target_dtype=self.dtype)
    ndim = self.rank
    pad_shape = []
    index_rank = index.rank

    # InplaceAdd doesn't support index with size 0, so unsqueeze operator is needed here
    if index_rank == 0:
        index = ge.Unsqueeze(index, axes=[0])

    # We shouldn't use shape = index._symsize(), because it will cause error in dynamic mode
    shape_index = ge.Shape(index, dtype=DataType.DT_INT32)
    shape_self = ge.Shape(self, dtype=DataType.DT_INT32)

    # Index's rank attribute will be None after dtype_prompt
    # So it must be placed after index.rank
    if index.dtype != DataType.DT_INT32:
        index = dtype_promote(index, target_dtype=DataType.DT_INT32)
        logger.warning_once("torch.ops.aten.index_add.default: "
                            "one int64 will be transferred into two int32 in the NPU kernel and cause error, "
                            "it was transferred to int32 automatically.")
    
    if dim < 0:
        dim += ndim
        logger.warning_once("torch.ops.aten.index_add.default: "
                            "operator in NPU doesn't support negative dim now, "
                            "it was transferred to positive value automatically.")

    # pad_shape[i] = self.shape[i], except pad_shape[dim] = index.shape[0]
    for i in range(ndim):
        if i != dim:
            cur_self_shape = ge.Gather(shape_self, i)
            pad_shape.append(cur_self_shape)
        else:
            dim_index_value = ge.Gather(shape_index, 0)
            pad_shape.append(dim_index_value)
    pad_size = ge.Pack(pad_shape, N=ndim, axis=0)
    source_broadcast = ge.BroadcastTo(source, pad_size)

    # InplaceIndexAdd is an inplace operator, it will cause precision problem if we use self as input
    new_input = ge.TensorMove(self)
    result = ge.InplaceIndexAdd(new_input, index, source_broadcast, alpha, axis=dim)
    return result


@register_fx_node_ge_converter(torch.ops.aten.index_add.out)
def conveter_aten_index_add_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.index_add.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.index_add.dimname)
def conveter_aten_index_add_dimname(
    self: Tensor,
    dim: str,
    index: Tensor,
    source: Tensor,
    *,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::index_add.dimname(Tensor self, str dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor"""
    raise RuntimeError("aten::index_add.dimname is not yet supported with named tensors. ")
