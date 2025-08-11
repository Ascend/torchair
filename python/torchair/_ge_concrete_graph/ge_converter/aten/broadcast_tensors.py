from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.broadcast_tensors.default)
def conveter_aten_broadcast_tensors_default(
    tensors: List[Tensor], meta_outputs: List[TensorSpec] = None
):
    """NB: aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.broadcast_tensors.default ge_converter is not implemented!")
