from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.split_dim.default)
def conveter_prims_split_dim_default(
    a: Tensor, dim: int, outer_length: Union[int, Tensor], meta_outputs: TensorSpec = None
):
    """NB: prims::split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.prims.split_dim.default ge_converter is not implemented!")
