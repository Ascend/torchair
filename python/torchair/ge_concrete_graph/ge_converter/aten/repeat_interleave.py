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


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor)
def conveter_aten_repeat_interleave_Tensor(
        repeats: Tensor,
        *,
        output_size: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_Tensor)
def conveter_aten_repeat_interleave_self_Tensor(
        self: Tensor,
        repeats: Tensor,
        dim: Optional[int] = None,
        *,
        output_size: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_int)
def conveter_aten_repeat_interleave_self_int(
        self: Tensor,
        repeats: Union[int, Tensor],
        dim: Optional[int] = None,
        *,
        output_size: Optional[int] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, int? output_size=None) -> Tensor """
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor_out)
def conveter_aten_repeat_interleave_Tensor_out(
        repeats: Tensor,
        *,
        output_size: Optional[int] = None,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::repeat_interleave.Tensor_out(Tensor repeats, *, int? output_size=None, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor_out ge converter is not implement!")


