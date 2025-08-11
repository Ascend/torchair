from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split.Tensor)
def conveter_aten_unsafe_split_Tensor(
    self: Tensor, split_size: Union[int, Tensor], dim: int = 0, meta_outputs: List[TensorSpec] = None
):
    """NB: aten::unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.unsafe_split.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split.Tensor_out)
def conveter_aten_unsafe_split_Tensor_out(
    self: Tensor,
    split_size: Union[int, Tensor],
    dim: int = 0,
    *,
    out: List[Tensor] = None
):
    """NB: aten::unsafe_split.Tensor_out(Tensor self, SymInt split_size, int dim=0, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten.unsafe_split.Tensor_out ge_converter is not implemented!")
