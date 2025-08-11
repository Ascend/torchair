from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.celu_.default)
def conveter_aten_celu__default(
    self: Tensor, alpha: Union[Number, Tensor] = 1.0, meta_outputs: TensorSpec = None
):
    """NB: aten::celu_(Tensor(a!) self, Scalar alpha=1.) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.celu_.default ge_converter is not implemented!")
