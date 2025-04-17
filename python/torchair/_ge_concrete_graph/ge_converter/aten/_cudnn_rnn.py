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


@register_fx_node_ge_converter(torch.ops.aten._cudnn_rnn.out)
def conveter_aten__cudnn_rnn_out(
    input: Tensor,
    weight: List[Tensor],
    weight_stride0: int,
    weight_buf: Optional[Tensor],
    hx: Tensor,
    cx: Optional[Tensor],
    mode: int,
    hidden_size: Union[int, Tensor],
    proj_size: Union[int, Tensor],
    num_layers: int,
    batch_first: bool,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_sizes: Union[List[int], Tensor],
    dropout_state: Optional[Tensor],
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    out3: Tensor = None,
    out4: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_cudnn_rnn.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))"""
    raise NotImplementedError("torch.ops.aten._cudnn_rnn.out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._cudnn_rnn.default)
def conveter_aten__cudnn_rnn_default(
    input: Tensor,
    weight: List[Tensor],
    weight_stride0: int,
    weight_buf: Optional[Tensor],
    hx: Tensor,
    cx: Optional[Tensor],
    mode: int,
    hidden_size: Union[int, Tensor],
    proj_size: Union[int, Tensor],
    num_layers: int,
    batch_first: bool,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_sizes: Union[List[int], Tensor],
    dropout_state: Optional[Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten._cudnn_rnn.default ge_converter is not implemented!")
