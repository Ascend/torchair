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
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.select.Dimname)
def conveter_aten_select_Dimname(
    self: Tensor, dim: str, index: int, meta_outputs: TensorSpec = None
):
    """NB: aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)"""
    raise RuntimeError(
        "torch.ops.aten.select.Dimname is redundant before pytorch 2.1.0,might be supported in future version.")


# TO DO: case2 will fail, fix view output case later
@declare_supported(
    [
        Support(F32(3, 4), dim=0, index=0),
        Support(F16(3, 4, 5), dim=1, index=2),
        Support(F16(3, 4, 5), dim=1, index=-1)
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.select.int)
def conveter_aten_select_int(
    self: Tensor, dim: int, index: Union[int, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)"""

    return ge.GatherV2(self, index, [dim], negative_index_support=True)