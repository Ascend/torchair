import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
from torchair.ge_concrete_graph.ge_graph import DataType
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


@register_fx_node_ge_converter(torch.ops.aten.sym_size.default)
def conveter_aten_sym_size_default(
        self: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::sym_size(Tensor self) -> SymInt[] """
    raise NotImplementedError(
        "torch.ops.aten.sym_size.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.sym_size)
def conveter_aten_sym_size_int(
        self: Tensor,
        dim: int,
        meta_outputs: Any = None):
    """ NB: aten::sym_size.int(Tensor self, int dim) -> SymInt """
    if isinstance(meta_outputs, int):
        return meta_outputs
    if isinstance(meta_outputs, torch.SymInt):
        try:
            return int(str(meta_outputs))
        except:
            pass  # Not static dim size
    shape = ge.Shape(self, dtype=DataType.DT_INT64)
    return ge.Gather(shape, dim)
