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


@register_fx_node_ge_converter(torch.ops.aten.mkldnn_rnn_layer.default)
def conveter_aten_mkldnn_rnn_layer_default(
    input: Tensor,
    weight0: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    weight3: Tensor,
    hx_: Tensor,
    cx_: Tensor,
    reverse: bool,
    batch_sizes: List[int],
    mode: int,
    hidden_size: int,
    num_layers: int,
    has_biases: bool,
    bidirectional: bool,
    batch_first: bool,
    train: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::mkldnn_rnn_layer(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) -> (Tensor, Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.mkldnn_rnn_layer.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.mkldnn_rnn_layer.out)
def conveter_aten_mkldnn_rnn_layer_out(
    input: Tensor,
    weight0: Tensor,
    weight1: Tensor,
    weight2: Tensor,
    weight3: Tensor,
    hx_: Tensor,
    cx_: Tensor,
    reverse: bool,
    batch_sizes: List[int],
    mode: int,
    hidden_size: int,
    num_layers: int,
    has_biases: bool,
    bidirectional: bool,
    batch_first: bool,
    train: bool,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    out3: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::mkldnn_rnn_layer.out(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))"""
    raise NotImplementedError("torch.ops.aten.mkldnn_rnn_layer.out ge_converter is not implemented!")
