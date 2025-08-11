from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.cat.default)
def conveter_prims_cat_default(
    tensors: List[Tensor], dim: int, meta_outputs: TensorSpec = None
):
    """NB: prims::cat(Tensor[] tensors, int dim) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.cat.default ge_converter is not implemented!")
