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
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
    torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@declare_supported([
    Support(F32(3, 4), F32(3), 1, -1),
    Support(F32(10, 22), F32(22), 0, 0),
])
@register_fx_node_ge_converter(torch.ops.aten.select_scatter.default)
def conveter_aten_select_scatter_default(
    self: Tensor,
    src: Tensor,
    dim: int,
    index: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor"""
    input_sizes = ge.Shape(self)
   
    index_ = ge.BroadcastTo(index, input_sizes)
    src_ = ge.BroadcastTo(ge.ExpandDims(src, dim), input_sizes)

    return ge.ScatterElements(self, index_, src_, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.select_scatter.out)
def conveter_aten_select_scatter_out(
    self: Tensor,
    src: Tensor,
    dim: int,
    index: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.select_scatter.out ge_converter is not supported!")
