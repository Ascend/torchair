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


@register_fx_node_ge_converter(torch.ops.aten.rnn_relu.input)
def conveter_aten_rnn_relu_input(
    input: Tensor,
    hx: Tensor,
    params: List[Tensor],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.rnn_relu.input ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rnn_relu.data)
def conveter_aten_rnn_relu_data(
    data: Tensor,
    batch_sizes: Tensor,
    hx: Tensor,
    params: List[Tensor],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.rnn_relu.data ge_converter is not implemented!")
