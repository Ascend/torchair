from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.set_.source_Tensor)
def conveter_aten_set__source_Tensor(
    self: Tensor, source: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.set_.source_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.set_.default)
def conveter_aten_set__default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::set_(Tensor(a!) self) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.set_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.set_.source_Tensor_storage_offset)
def conveter_aten_set__source_Tensor_storage_offset(
    self: Tensor,
    source: Tensor,
    storage_offset: Union[int, Tensor],
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor] = [],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.set_.source_Tensor_storage_offset ge_converter is not implemented!")
