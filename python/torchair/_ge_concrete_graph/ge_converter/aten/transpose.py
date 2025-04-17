from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
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



@declare_supported(
    [
        Support(F32(2, 3), 0, 1),
        Support(I8(2, 3), 0, 1),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.transpose.int)
def conveter_aten_transpose_int(
    self: Tensor, dim0: int, dim1: int, meta_outputs: TensorSpec = None
):
    """NB: aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""
    perm_list = [i for i in range(self.rank)]
    perm_list[dim0], perm_list[dim1] = perm_list[dim1], perm_list[dim0]
    perm_list = dtype_promote(perm_list, target_dtype=DataType.DT_INT64)
    return ge.Transpose(self, perm_list)


@register_fx_node_ge_converter(torch.ops.aten.transpose.Dimname)
def conveter_aten_transpose_Dimname(
    self: Tensor, dim0: str, dim1: str, meta_outputs: TensorSpec = None
):
    """NB: aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.transpose.Dimname ge_converter is not implemented!")
