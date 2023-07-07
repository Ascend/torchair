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


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.default)
def conveter_aten_diag_embed_default(
        self: Tensor,
        offset: int = 0,
        dim1: int = -2,
        dim2: int = -1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor """
    raise NotImplementedError("torch.ops.aten.diag_embed.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.diag_embed.out)
def conveter_aten_diag_embed_out(
        self: Tensor,
        offset: int = 0,
        dim1: int = -2,
        dim2: int = -1,
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::diag_embed.out(Tensor self, int offset=0, int dim1=-2, int dim2=-1, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.diag_embed.out ge converter is not implement!")


