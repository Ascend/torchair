from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.nan_to_num_.default)
def conveter_aten_nan_to_num__default(
    self: Tensor,
    nan: Optional[float] = None,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.nan_to_num_.default ge_converter is not implemented!")
