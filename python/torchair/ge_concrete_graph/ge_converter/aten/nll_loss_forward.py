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


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_forward.default)
def conveter_aten_nll_loss_forward_default(
        self: Tensor,
        target: Tensor,
        weight: Optional[Tensor],
        reduction: int,
        ignore_index: Union[int, Tensor],
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight) """
    raise NotImplementedError("torch.ops.aten.nll_loss_forward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.nll_loss_forward.output)
def conveter_aten_nll_loss_forward_output(
        self: Tensor,
        target: Tensor,
        weight: Optional[Tensor],
        reduction: int,
        ignore_index: Union[int, Tensor],
        *,
        output: Tensor = None,
        total_weight: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!)) """
    raise NotImplementedError("torch.ops.aten.nll_loss_forward.output ge converter is not implement!")


