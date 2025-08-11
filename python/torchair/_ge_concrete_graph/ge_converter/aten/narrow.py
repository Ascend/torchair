from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.narrow.default)
def conveter_aten_narrow_default(
    self: Tensor,
    dim: int,
    start: Union[int, Tensor],
    length: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.narrow.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.narrow.Tensor)
def conveter_aten_narrow_Tensor(
    self: Tensor,
    dim: int,
    start: Tensor,
    length: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.narrow.Tensor ge_converter is not implemented!")
