from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.default)
def conveter_aten_logcumsumexp_default(
    self: Tensor, dim: int, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp(Tensor self, int dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname)
def conveter_aten_logcumsumexp_dimname(
    self: Tensor, dim: str, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.dimname(Tensor self, str dim) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.dimname_out)
def conveter_aten_logcumsumexp_dimname_out(
    self: Tensor, dim: str, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.dimname_out(Tensor self, str dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.dimname_out ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.logcumsumexp.out)
def conveter_aten_logcumsumexp_out(
    self: Tensor, dim: int, *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.logcumsumexp.out ge_converter is not implemented!")
