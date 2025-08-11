from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.resize.default)
def conveter_prims_resize_default(
    a: Tensor, shape: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: prims::resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.prims.resize.default ge_converter is not implemented!")
