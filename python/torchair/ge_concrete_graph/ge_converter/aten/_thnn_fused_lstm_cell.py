import torch
from torchair.ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor
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


@register_fx_node_ge_converter(torch.ops.aten._thnn_fused_lstm_cell.default)
def conveter_aten__thnn_fused_lstm_cell_default(
        input_gates: Tensor,
        hidden_gates: Tensor,
        cx: Tensor,
        input_bias: Optional[Tensor] = None,
        hidden_bias: Optional[Tensor] = None,
        meta_outputs: Any = None):
    """ NB: aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten._thnn_fused_lstm_cell.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten._thnn_fused_lstm_cell.out)
def conveter_aten__thnn_fused_lstm_cell_out(
        input_gates: Tensor,
        hidden_gates: Tensor,
        cx: Tensor,
        input_bias: Optional[Tensor] = None,
        hidden_bias: Optional[Tensor] = None,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::_thnn_fused_lstm_cell.out(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!)) """
    raise NotImplementedError("torch.ops.aten._thnn_fused_lstm_cell.out ge converter is not implement!")


