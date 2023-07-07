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


@register_fx_node_ge_converter(torch.ops.aten.transpose.int)
def conveter_aten_transpose_int(
        self: Tensor,
        dim0: int,
        dim1: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a) """
    perm_list = [i for i in range(self.rank)]
    perm_list[dim0], perm_list[dim1] = perm_list[dim1], perm_list[dim0]
    return ge.Transpose(self, perm_list)


@register_fx_node_ge_converter(torch.ops.aten.transpose.Dimname)
def conveter_aten_transpose_Dimname(
        self: Tensor,
        dim0: str,
        dim1: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> Tensor(a) """
    raise NotImplementedError("torch.ops.aten.transpose.Dimname ge converter is not implement!")


