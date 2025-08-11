from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.pixel_shuffle.default)
def conveter_aten_pixel_shuffle_default(
    self: Tensor, upscale_factor: int, meta_outputs: TensorSpec = None
):
    """NB: aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.pixel_shuffle.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.pixel_shuffle.out)
def conveter_aten_pixel_shuffle_out(
    self: Tensor, upscale_factor: int, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::pixel_shuffle.out(Tensor self, int upscale_factor, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.pixel_shuffle.out ge_converter is not implemented!")
