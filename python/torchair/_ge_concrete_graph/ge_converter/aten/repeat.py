from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.repeat.default)
def conveter_aten_repeat_default(
    self: Tensor, repeats: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::repeat(Tensor self, SymInt[] repeats) -> Tensor"""
    # TO DO: add check between self.rank and repeats length
    return ge.Tile(self, repeats)


@register_fx_node_ge_converter(torch.ops.aten.repeat.out)
def conveter_aten_repeat_out(
    self: Tensor,
    repeats: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat.out(Tensor self, SymInt[] repeats, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.repeat.out ge_converter is not implemented!")
