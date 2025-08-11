from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.threshold_.default)
def conveter_aten_threshold__default(
    self: Tensor,
    threshold: Union[Number, Tensor],
    value: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.threshold_.default ge_converter is not implemented!")
