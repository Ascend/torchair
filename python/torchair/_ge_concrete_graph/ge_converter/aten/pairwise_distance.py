from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.pairwise_distance.default)
def conveter_aten_pairwise_distance_default(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    eps: float = 1e-06,
    keepdim: bool = False,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.pairwise_distance.default ge_converter is not implemented!")
