from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.quantized_lstm.input_legacy ge_converter is not implemented!")


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
    meta_outputs: TensorSpec = None
):
    """NB: aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.quantized_lstm.data_legacy ge_converter is not implemented!")
