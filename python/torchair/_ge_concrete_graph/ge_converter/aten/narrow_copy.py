from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.default)
def conveter_aten_narrow_copy_default(
    self: Tensor,
    dim: int,
    start: Union[int, Tensor],
    length: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.narrow_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.narrow_copy.out)
def conveter_aten_narrow_copy_out(
    self: Tensor,
    dim: int,
    start: Union[int, Tensor],
    length: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.narrow_copy.out ge_converter is not implemented!")
