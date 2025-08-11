from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(3, 4), F32(3), 1, -1),
    Support(F32(10, 22), F32(22), 0, 0),
])
@register_fx_node_ge_converter(torch.ops.aten.select_scatter.default)
def conveter_aten_select_scatter_default(
    self: Tensor,
    src: Tensor,
    dim: int,
    index: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor"""
    input_sizes = ge.Shape(self)
   
    index_ = ge.BroadcastTo(index, input_sizes)
    src_ = ge.BroadcastTo(ge.ExpandDims(src, dim), input_sizes)

    return ge.ScatterElements(self, index_, src_, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.select_scatter.out)
def conveter_aten_select_scatter_out(
    self: Tensor,
    src: Tensor,
    dim: int,
    index: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.select_scatter.out ge_converter is not supported!")
