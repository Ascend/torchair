from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.amin.default)
def conveter_aten_amin_default(
    self: Tensor, dim: List[int] = (), keepdim: bool = False, meta_outputs: TensorSpec = None
):
    """NB: aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.amin.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.amin.out)
def conveter_aten_amin_out(
    self: Tensor,
    dim: List[int] = (),
    keepdim: bool = False,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.amin.out ge_converter is not implemented!")
