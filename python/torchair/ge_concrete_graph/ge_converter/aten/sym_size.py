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
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import DataType, Tensor, TensorSpec
from torchair.ge_concrete_graph.utils import force_op_unknown_shape


@register_fx_node_ge_converter(torch.ops.aten.sym_size.default)
def conveter_aten_sym_size_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::sym_size(Tensor self) -> SymInt[]"""
    raise NotImplementedError("torch.ops.aten.sym_size.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.sym_size)
def conveter_aten_sym_size_int(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::sym_size.int(Tensor self, int dim) -> SymInt"""
    if isinstance(meta_outputs._meta, int):
        return meta_outputs._meta
    if isinstance(meta_outputs._meta, torch.SymInt):
        try:
            return int(str(meta_outputs._meta))
        except:
            pass  # Not static dim size
    shape = ge.Shape(self, dtype=DataType.DT_INT64)
    # force unknown shape with ge.Gather when parse symsize
    return force_op_unknown_shape(ge.Gather(shape, dim))


@register_fx_node_ge_converter(torch.ops.aten.sym_numel)
def convert_aten_sym_numel(
    self,
    out: Tensor = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::sym_size.sym_numel(Tensor self) -> SymInt"""
    return ge.Cast(ge.ReduceProdD(ge.Shape(self), axes=[0]), dst_type=DataType.DT_INT64)

