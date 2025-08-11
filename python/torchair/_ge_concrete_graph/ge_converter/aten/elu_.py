from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.elu_.default)
def conveter_aten_elu__default(
    self: Tensor,
    alpha: Union[Number, Tensor] = 1,
    scale: Union[Number, Tensor] = 1,
    input_scale: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.elu_.default ge_converter is not implemented!")
