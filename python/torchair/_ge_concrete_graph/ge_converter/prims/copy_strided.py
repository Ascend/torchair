from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.copy_strided.default)
def conveter_prims_copy_strided_default(
    a: Tensor, stride: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: prims::copy_strided(Tensor a, SymInt[] stride) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.copy_strided.default ge_converter is not implemented!")
