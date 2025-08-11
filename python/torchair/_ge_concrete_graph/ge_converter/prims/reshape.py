from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.reshape.default)
def conveter_prims_reshape_default(
    a: Tensor, shape: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: prims::reshape(Tensor a, SymInt[] shape) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.reshape.default ge_converter is not implemented!")
