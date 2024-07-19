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
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote

@declare_supported([
    Support(F32(2, 5, 2, 2), [2, 2]),
])
@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d.default)
def conveter_aten__adaptive_avg_pool2d_default(
    self: Tensor, output_size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor"""
    if isinstance(output_size, Tensor):
        raise NotImplementedError("torch.ops.aten._adaptive_avg_pool2d.default with output_size in tensor ge_converter "
                                  "is not implemented when output_size is tensor")
    return ge.AdaptiveAvgPool2d(self, output_size=output_size)


@register_fx_node_ge_converter(torch.ops.aten._adaptive_avg_pool2d.out)
def conveter_aten__adaptive_avg_pool2d_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._adaptive_avg_pool2d.out ge_converter is not supported!")