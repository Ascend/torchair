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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support, T, BF16
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(3, 5), 0, T([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.int64), F32(2, 5)),
        Support(F32(10,), 0, T([2, 4, 8], dtype=torch.int64), F32(20,)),
        Support(F16(20, 21, 30), 0, I64(20, 20, 15, value_range=(0, 20)), F16(20, 20, 15)),
        Support(I32(5, 5, 5), 1, I64(5, 5, 2, value_range=(0, 5)), I32(5, 5, 2)),
        Support(I8(5, 5, 5), 2, I64(5, 5, 2, value_range=(0, 5)), I8(5, 5, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.scatter_add.default)
def conveter_aten_scatter_add_default(
    self: Tensor, dim: int, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor"""
    # index is empty tensor, return self
    for elem in index.symsize:
        if elem == 0:
            return self
    if dim < 0:
        dim += self.rank
    last_dim = -1 + self.rank

    # input dim is not last dim
    if dim != last_dim:
        perm = []
        for i in range(self.rank):
            perm.append(i)
        perm[dim], perm[last_dim] = perm[last_dim], perm[dim]
        perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
        transpose_self = ge.Transpose(self, perm)
        transpose_idx = ge.Transpose(index, perm)
        transpose_src = ge.Transpose(src, perm)
        if use_scatterelements_v2(self, dim, index):
            res = ge.ScatterElementsV2(transpose_self, transpose_idx, transpose_src, axis=last_dim, reduction='add')
            return ge.Transpose(res, perm)
        else:
            return ge.ScatterElements(self, index, src, axis=dim, reduction='add')

    if use_scatterelements_v2(self, dim, index):
        return ge.ScatterElementsV2(self, index, src, axis=dim, reduction='add')
    else:
        return ge.ScatterElements(self, index, src, axis=dim, reduction='add')


def can_combineaxis_v2(data, dim, index):
    for i in range(data.rank):
        if data.symsize[i] != index.symsize[i]:
            if i != dim and i != 0 and i != data.rank - 1:
                return False
    return True


def use_scatterelements_v2(data, dim, index):
    # only these dtype can use V2
    if data.dtype not in [DataType.DT_FLOAT, DataType.DT_FLOAT16, DataType.DT_BF16, DataType.DT_INT32, 
                          DataType.DT_INT8, DataType.DT_UINT8]:
        return False
    if dim != data.rank - 1 and dim != 0 :
        return False
    if not can_combineaxis_v2(data, dim, index):
        return False
    return True


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.out)
def conveter_aten_scatter_add_out(
    self: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.scatter_add.out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.scatter_add.dimname)
def conveter_aten_scatter_add_dimname(
    self: Tensor, dim: str, index: Tensor, src: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> Tensor"""
    raise RuntimeError("aten::scatter_add.dimname is not yet supported with named tensors. ")
