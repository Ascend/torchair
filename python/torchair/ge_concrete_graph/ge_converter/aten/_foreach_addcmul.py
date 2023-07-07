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


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Scalar)
def conveter_aten__foreach_addcmul_Scalar(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        value: Union[Number, Tensor] = 1,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.Scalar ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.ScalarList)
def conveter_aten__foreach_addcmul_ScalarList(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Union[List[Number], Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.ScalarList ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Tensor)
def conveter_aten__foreach_addcmul_Tensor(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Tensor,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[] """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.Tensor ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Scalar_out)
def conveter_aten__foreach_addcmul_Scalar_out(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        value: Union[Number, Tensor] = 1,
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.Scalar_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.ScalarList_out)
def conveter_aten__foreach_addcmul_ScalarList_out(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Union[List[Number], Tensor],
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.ScalarList_out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._foreach_addcmul.Tensor_out)
def conveter_aten__foreach_addcmul_Tensor_out(
        self: List[Tensor],
        tensor1: List[Tensor],
        tensor2: List[Tensor],
        scalars: Tensor,
        *,
        out: List[Tensor] = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_foreach_addcmul.Tensor_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars, *, Tensor(a!)[] out) -> () """
    raise NotImplementedError("torch.ops.aten._foreach_addcmul.Tensor_out ge converter is not implement!")


