from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.default)
def conveter_aten_meshgrid_default(tensors: List[Tensor], meta_outputs: List[TensorSpec] = None):
    """NB: aten::meshgrid(Tensor[] tensors) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.meshgrid.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.meshgrid.indexing)
def conveter_aten_meshgrid_indexing(
    tensors: List[Tensor], *, indexing: str, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.meshgrid.indexing ge_converter is not implemented!")
