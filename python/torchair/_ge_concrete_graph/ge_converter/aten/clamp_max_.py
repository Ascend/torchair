from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.clamp_max_.default)
def conveter_aten_clamp_max__default(
    self: Tensor, max: Union[Number, Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max_.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.clamp_max_.Tensor)
def conveter_aten_clamp_max__Tensor(
    self: Tensor, max: Tensor, meta_outputs: TensorSpec = None
):
    """NB: aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.clamp_max_.Tensor ge_converter is not implemented!")
