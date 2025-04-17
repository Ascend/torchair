from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported(
    [
        Support(F32(2, 2, 2), F32(2, 2, 2)),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.bmm.default)
def conveter_aten_bmm_default(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::bmm(Tensor self, Tensor mat2) -> Tensor"""
    if self.dtype == DataType.DT_INT8 or mat2.dtype == DataType.DT_INT8:
        raise RuntimeError("torch.ops.aten.bmm.default ge_converter is not support int8 dtype!")
    self, mat2 = dtype_promote(self, mat2, target_dtype=meta_outputs.dtype)
    return ge.BatchMatMul(self, mat2)


@register_fx_node_ge_converter(torch.ops.aten.bmm.out)
def conveter_aten_bmm_out(
    self: Tensor, mat2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.bmm.out ge_converter is not implemented!")
