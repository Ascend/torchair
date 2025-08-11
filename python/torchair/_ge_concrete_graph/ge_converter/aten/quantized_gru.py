from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.quantized_gru.input_legacy)
def conveter_aten_quantized_gru_input_legacy(
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
    """NB: aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.quantized_gru.input_legacy ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.quantized_gru.data_legacy)
def conveter_aten_quantized_gru_data_legacy(
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
    """NB: aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.aten.quantized_gru.data_legacy ge_converter is not implemented!")
