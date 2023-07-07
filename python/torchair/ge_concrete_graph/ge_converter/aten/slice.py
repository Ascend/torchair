import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor
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


@register_fx_node_ge_converter(torch.ops.aten.slice.Tensor)
def conveter_aten_slice_Tensor(
        self: Tensor,
        dim: int = 0,
        start: Optional[Union[int, Tensor]] = None,
        end: Optional[Union[int, Tensor]] = None,
        step: Union[int, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a) """
    offsets = [0 for _ in range(self.rank)]
    size = [-1 for _ in range(self.rank)]
    offsets[dim] = start

    if (end - start) % step != 0:
        raise NotImplementedError("torch.ops.aten.slice.Tensor unsupported param!")
    if end != 9223372036854775807:
        size[dim] = (end - start) // step
    return ge.Slice(self, offsets, size)

@register_fx_node_ge_converter(torch.ops.aten.slice.str)
def conveter_aten_slice_str(
        string: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::slice.str(str string, int? start=None, int? end=None, int step=1) -> str """
    raise NotImplementedError("torch.ops.aten.slice.str ge converter is not implement!")


