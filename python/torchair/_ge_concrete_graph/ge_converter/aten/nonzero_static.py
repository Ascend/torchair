from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nonzero_static.default)
def conveter_aten_nonzero_static_default(
    self: Tensor, *, size: int, fill_value: int = -1, meta_outputs: TensorSpec = None
):
    """NB: aten::nonzero_static(Tensor self, *, int size, int fill_value=-1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.nonzero_static.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.nonzero_static.out)
def conveter_aten_nonzero_static_out(
    self: Tensor,
    *,
    size: int,
    fill_value: int = -1,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nonzero_static.out(Tensor self, *, int size, int fill_value=-1, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nonzero_static.out ge_converter is not implemented!")
