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


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor)
def conveter_aten_index_Tensor(
        self: Tensor,
        indices: List[Optional[Tensor]],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor_out)
def conveter_aten_index_Tensor_out(
        self: Tensor,
        indices: List[Optional[Tensor]],
        *,
        out: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) -> Tensor(a!) """
    raise NotImplementedError("torch.ops.aten.index.Tensor_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.Tensor_hacked_twin)
def conveter_aten_index_Tensor_hacked_twin(
        self: Tensor,
        indices: List[Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.Tensor_hacked_twin(Tensor self, Tensor[] indices) -> Tensor """
    raise NotImplementedError("torch.ops.aten.index.Tensor_hacked_twin ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.str)
def conveter_aten_index_str(
        self: str,
        substr: str,
        start: int = 0,
        end: int = -1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.str(str self, str substr, int start=0, int end=-1) -> int """
    raise NotImplementedError("torch.ops.aten.index.str ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_int)
def conveter_aten_index_list_int(
        self: List[int],
        el: int,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.list_int(int[] self, int el) -> int """
    raise NotImplementedError("torch.ops.aten.index.list_int ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_float)
def conveter_aten_index_list_float(
        self: List[float],
        el: float,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.list_float(float[] self, float el) -> int """
    raise NotImplementedError("torch.ops.aten.index.list_float ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_bool)
def conveter_aten_index_list_bool(
        self: List[bool],
        el: bool,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.list_bool(bool[] self, bool el) -> int """
    raise NotImplementedError("torch.ops.aten.index.list_bool ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_Tensor)
def conveter_aten_index_list_Tensor(
        self: List[Tensor],
        el: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.list_Tensor(Tensor[] self, Tensor el) -> int """
    raise NotImplementedError("torch.ops.aten.index.list_Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.index.list_str)
def conveter_aten_index_list_str(
        self: List[str],
        el: str,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::index.list_str(str[] self, str el) -> int """
    raise NotImplementedError("torch.ops.aten.index.list_str ge converter is not implement!")


