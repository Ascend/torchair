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
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import DataType, Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.aten.sym_size.default)
def conveter_aten_sym_size_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sym_size(Tensor self) -> SymInt[]"""
    raise NotImplementedError("torch.ops.aten.sym_size.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sym_size)
def conveter_aten_sym_size_int(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::sym_size.int(Tensor self, int dim) -> SymInt"""
    if isinstance(meta_outputs._meta, int):
        return meta_outputs
    if isinstance(meta_outputs._meta, torch.SymInt):
        try:
            return int(str(meta_outputs._meta))
        except:
            pass  # Not static dim size
    shape = ge.Shape(self, dtype=DataType.DT_INT64)
    return ge.Gather(shape, dim)
