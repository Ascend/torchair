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


@register_fx_node_ge_converter(torch.ops.aten._thnn_fused_lstm_cell_backward_impl.out)
def conveter_aten__thnn_fused_lstm_cell_backward_impl_out(
        grad_hy: Optional[Tensor],
        grad_cy: Optional[Tensor],
        cx: Tensor,
        cy: Tensor,
        workspace: Tensor,
        has_bias: bool,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_thnn_fused_lstm_cell_backward_impl.out(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten._thnn_fused_lstm_cell_backward_impl.out ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._thnn_fused_lstm_cell_backward_impl.default)
def conveter_aten__thnn_fused_lstm_cell_backward_impl_default(
        grad_hy: Optional[Tensor],
        grad_cy: Optional[Tensor],
        cx: Tensor,
        cy: Tensor,
        workspace: Tensor,
        has_bias: bool,
        meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    """ NB: aten::_thnn_fused_lstm_cell_backward_impl(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._thnn_fused_lstm_cell_backward_impl.default ge converter is not implement!")


