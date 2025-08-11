from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.size.int)
def conveter_aten_size_int(self: Tensor, dim: int, meta_outputs: TensorSpec = None):
    """NB: aten::size.int(Tensor self, int dim) -> int"""
    raise NotImplementedError("torch.ops.aten.size.int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.size.Dimname)
def conveter_aten_size_Dimname(self: Tensor, dim: str, meta_outputs: TensorSpec = None):
    """NB: aten::size.Dimname(Tensor self, str dim) -> int"""
    raise NotImplementedError("torch.ops.aten.size.Dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.size.default)
def conveter_aten_size_default(self: Tensor, meta_outputs: TensorSpec = None):
    """NB: aten::size(Tensor self) -> int[]"""
    raise NotImplementedError("torch.ops.aten.size.default ge_converter is not implemented!")
