from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diagonal_scatter.default)
def conveter_aten_diagonal_scatter_default(
    self: Tensor,
    src: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diagonal_scatter.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_scatter.out)
def conveter_aten_diagonal_scatter_out(
    self: Tensor,
    src: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diagonal_scatter.out(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diagonal_scatter.out ge_converter is not implemented!")
