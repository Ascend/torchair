from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.gt.default)
def conveter_prims_gt_default(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
    """NB: prims::gt(Tensor self, Tensor other) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.gt.default ge_converter is not implemented!")
