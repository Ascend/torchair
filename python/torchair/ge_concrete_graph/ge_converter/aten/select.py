import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.fx2ge_converter import register_testcase
from torchair.ge_concrete_graph.testing_utils import *
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import dtype_promote
from torch import contiguous_format, Generator, inf, memory_format, strided
from torchair.ge_concrete_graph import ge_apis as ge
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from torch.types import (
    _bool,
    _complex,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Device,
    Number,
    SymInt,
)


@register_fx_node_ge_converter(torch.ops.aten.select.Dimname)
def conveter_aten_select_Dimname(
        self: Tensor,
        dim: str,
        index: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.select.Dimname ge converter is not implement!")


# TODO: case2 will fail, fix view output case later
@register_testcase([
    TestInput(F32(3, 4), dim=0, index=0),
    TestInput(F16(3, 4, 5), dim=1, index=2),
])
@register_fx_node_ge_converter(torch.ops.aten.select.int)
def conveter_aten_select_int(
        self: Tensor,
        dim: int,
        index: Union[int, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a) """
    if isinstance(index, Tensor):
        raise NotImplementedError("torch.ops.aten.select.int ge converter is not implement!")

    offsets = [0 for _ in range(self.rank)]
    size = [-1 for _ in range(self.rank)]
    offsets[dim] = index
    size[dim] = 1
    slice = ge.Slice(self, offsets, size)
    return ge.Squeeze(slice, axis=[dim])

