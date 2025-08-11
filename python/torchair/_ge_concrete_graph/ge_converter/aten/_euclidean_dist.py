from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._euclidean_dist.default)
def conveter_aten__euclidean_dist_default(
    x1: Tensor, x2: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._euclidean_dist.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._euclidean_dist.out)
def conveter_aten__euclidean_dist_out(
    x1: Tensor, x2: Tensor, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::_euclidean_dist.out(Tensor x1, Tensor x2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._euclidean_dist.out ge_converter is not implemented!")
