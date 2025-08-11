from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.lstm.input)
def conveter_aten_lstm_input(
    input: Tensor,
    hx: List[Tensor],
    params: List[Tensor],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.lstm.input ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.lstm.data)
def conveter_aten_lstm_data(
    data: Tensor,
    batch_sizes: Tensor,
    hx: List[Tensor],
    params: List[Tensor],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.lstm.data ge_converter is not implemented!")
