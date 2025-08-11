from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.dropout.default)
def conveter_aten_dropout_default(
    input: Tensor, p: float, train: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::dropout(Tensor input, float p, bool train) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.dropout.default ge_converter is not implemented!")
