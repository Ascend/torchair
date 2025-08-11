from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diagonal_copy.default)
def conveter_aten_diagonal_copy_default(
    self: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diagonal_copy.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_copy.out)
def conveter_aten_diagonal_copy_out(
    self: Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diagonal_copy.out ge_converter is not implemented!")
