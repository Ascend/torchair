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


@register_fx_node_ge_converter(torch.ops.aten.mkldnn_rnn_layer_backward.default)
def conveter_aten_mkldnn_rnn_layer_backward_default(
        input: Tensor,
        weight1: Tensor,
        weight2: Tensor,
        weight3: Tensor,
        weight4: Tensor,
        hx_: Tensor,
        cx_tmp: Tensor,
        output: Tensor,
        hy_: Tensor,
        cy_: Tensor,
        grad_output: Optional[Tensor],
        grad_hy: Optional[Tensor],
        grad_cy: Optional[Tensor],
        reverse: bool,
        mode: int,
        hidden_size: int,
        num_layers: int,
        has_biases: bool,
        train: bool,
        bidirectional: bool,
        batch_sizes: List[int],
        batch_first: bool,
        workspace: Tensor,
        meta_outputs: Any = None):
    """ NB: aten::mkldnn_rnn_layer_backward(Tensor input, Tensor weight1, Tensor weight2, Tensor weight3, Tensor weight4, Tensor hx_, Tensor cx_tmp, Tensor output, Tensor hy_, Tensor cy_, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, bool reverse, int mode, int hidden_size, int num_layers, bool has_biases, bool train, bool bidirectional, int[] batch_sizes, bool batch_first, Tensor workspace) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) """
    raise NotImplementedError("torch.ops.aten.mkldnn_rnn_layer_backward.default ge converter is not implement!")


@register_fx_node_ge_converter(torch.ops.aten.mkldnn_rnn_layer_backward.out)
def conveter_aten_mkldnn_rnn_layer_backward_out(
        input: Tensor,
        weight1: Tensor,
        weight2: Tensor,
        weight3: Tensor,
        weight4: Tensor,
        hx_: Tensor,
        cx_tmp: Tensor,
        output: Tensor,
        hy_: Tensor,
        cy_: Tensor,
        grad_output: Optional[Tensor],
        grad_hy: Optional[Tensor],
        grad_cy: Optional[Tensor],
        reverse: bool,
        mode: int,
        hidden_size: int,
        num_layers: int,
        has_biases: bool,
        train: bool,
        bidirectional: bool,
        batch_sizes: List[int],
        batch_first: bool,
        workspace: Tensor,
        *,
        out0: Tensor = None,
        out1: Tensor = None,
        out2: Tensor = None,
        out3: Tensor = None,
        out4: Tensor = None,
        out5: Tensor = None,
        out6: Tensor = None,
        meta_outputs: Any = None):
    """ NB: aten::mkldnn_rnn_layer_backward.out(Tensor input, Tensor weight1, Tensor weight2, Tensor weight3, Tensor weight4, Tensor hx_, Tensor cx_tmp, Tensor output, Tensor hy_, Tensor cy_, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, bool reverse, int mode, int hidden_size, int num_layers, bool has_biases, bool train, bool bidirectional, int[] batch_sizes, bool batch_first, Tensor workspace, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4, Tensor(f!) out5, Tensor(g!) out6) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!)) """
    raise NotImplementedError("torch.ops.aten.mkldnn_rnn_layer_backward.out ge converter is not implement!")


