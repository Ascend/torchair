from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.log_normal.default)
def conveter_aten_log_normal_default(
    self: Tensor,
    mean: float = 1.0,
    std: float = 2.0,
    *,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::log_normal(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.log_normal.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.log_normal.out)
def conveter_aten_log_normal_out(
    self: Tensor,
    mean: float = 1.0,
    std: float = 2.0,
    *,
    generator: Optional[Generator] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::log_normal.out(Tensor self, float mean=1., float std=2., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.log_normal.out ge_converter is not implemented!")
