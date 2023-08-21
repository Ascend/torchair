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
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, \
    Support
from torchair.ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.aten.select.Dimname)
def conveter_aten_select_Dimname(
    self: Tensor, dim: str, index: int, meta_outputs: TensorSpec = None
):
    """NB: aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.select.Dimname ge_converter is not implemented!")


# TO DO: case2 will fail, fix view output case later
@declare_supported(
    [
        Support(F32(3, 4), dim=0, index=0),
        Support(F16(3, 4, 5), dim=1, index=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.select.int)
def conveter_aten_select_int(
    self: Tensor, dim: int, index: Union[int, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)"""
    if isinstance(index, Tensor):
        raise NotImplementedError("torch.ops.aten.select.int ge_converter is not implemented!")

    offsets = [0 for _ in range(self.rank)]
    size = [-1 for _ in range(self.rank)]
    offsets[dim] = index
    size[dim] = 1
    slice = ge.Slice(self, offsets, size)
    return ge.Squeeze(slice, axis=[dim])
