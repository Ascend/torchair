from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.alpha_dropout.default)
def conveter_aten_alpha_dropout_default(
    input: Tensor, p: float, train: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.alpha_dropout.default ge_converter is not implemented!")
