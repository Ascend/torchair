from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.roll.default)
def conveter_aten_roll_default(
    self: Tensor,
    shifts: Union[List[int], Tensor],
    dims: List[int] = [],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.roll.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.roll.out)
def conveter_aten_roll_out(
    self: Tensor,
    shifts: Union[List[int], Tensor],
    dims: List[int] = [],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::roll.out(Tensor self, SymInt[1] shifts, int[1] dims=[], *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.roll.out ge_converter is not implemented!")
