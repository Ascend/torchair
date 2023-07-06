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


@register_fx_node_ge_converter(torch.ops.aten.quantized_lstm.input_legacy)
def conveter_aten_quantized_lstm_input_legacy(
        input: Tensor,
        hx: List[Tensor],
        params: List[Tensor],
        has_biases: bool,
        num_layers: int,
        dropout: float,
        train: bool,
        bidirectional: bool,
        batch_first: bool,
        *,
        dtype: Optional[int] = None,
        use_dynamic: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.quantized_lstm.input_legacy ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.quantized_lstm.data_legacy)
def conveter_aten_quantized_lstm_data_legacy(
        data: Tensor,
        batch_sizes: Tensor,
        hx: List[Tensor],
        params: List[Tensor],
        has_biases: bool,
        num_layers: int,
        dropout: float,
        train: bool,
        bidirectional: bool,
        *,
        dtype: Optional[int] = None,
        use_dynamic: bool = False,
        meta_outputs: Any = None):
    """ NB: aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.quantized_lstm.data_legacy ge converter is not implement!")


