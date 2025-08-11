from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.default)
def conveter_aten_unsafe_split_with_sizes_default(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]"""
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unsafe_split_with_sizes.out)
def conveter_aten_unsafe_split_with_sizes_out(
    self: Tensor,
    split_sizes: Union[List[int], Tensor],
    dim: int = 0,
    *,
    out: List[Tensor] = None
):
    """NB: aten::unsafe_split_with_sizes.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"""
    raise NotImplementedError("torch.ops.aten.unsafe_split_with_sizes.out ge_converter is not implemented!")
