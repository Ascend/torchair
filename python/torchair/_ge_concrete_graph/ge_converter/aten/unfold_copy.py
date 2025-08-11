from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.default)
def conveter_aten_unfold_copy_default(
    self: Tensor, dimension: int, size: int, step: int, meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.unfold_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unfold_copy.out)
def conveter_aten_unfold_copy_out(
    self: Tensor,
    dimension: int,
    size: int,
    step: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.unfold_copy.out ge_converter is not implemented!")
