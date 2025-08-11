from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rnn_tanh.input)
def conveter_aten_rnn_tanh_input(
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
    """NB: aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.rnn_tanh.input ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rnn_tanh.data)
def conveter_aten_rnn_tanh_data(
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
    """NB: aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.rnn_tanh.data ge_converter is not implemented!")
