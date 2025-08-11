from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.renorm_.default)
def conveter_aten_renorm__default(
    self: Tensor,
    p: Union[Number, Tensor],
    dim: int,
    maxnorm: Union[Number, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.renorm_.default ge_converter is not implemented!")
