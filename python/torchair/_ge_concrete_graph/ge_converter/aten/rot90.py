from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rot90.default)
def conveter_aten_rot90_default(
    self: Tensor, k: int = 1, dims: List[int] = [0, 1], meta_outputs: TensorSpec = None
):
    """NB: aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rot90.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.rot90.out)
def conveter_aten_rot90_out(
    self: Tensor,
    k: int = 1,
    dims: List[int] = [0, 1],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::rot90.out(Tensor self, int k=1, int[] dims=[0, 1], *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.rot90.out ge_converter is not implemented!")
