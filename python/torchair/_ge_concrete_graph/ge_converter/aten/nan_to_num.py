from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num.default)
def conveter_aten_nan_to_num_default(
    self: Tensor,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor"""
    return ge.NanToNum(self, nan=nan, posinf=posinf, neginf=neginf)


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num.out)
def conveter_aten_nan_to_num_out(
    self: Tensor,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)"""
    out = ge.NanToNum(self, nan=nan, posinf=posinf, neginf=neginf)
    return out
