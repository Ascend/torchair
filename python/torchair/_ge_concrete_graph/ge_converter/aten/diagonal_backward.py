from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.diagonal_backward.default)
def conveter_aten_diagonal_backward_default(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    offset: int,
    dim1: int,
    dim2: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.diagonal_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.diagonal_backward.out)
def conveter_aten_diagonal_backward_out(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    offset: int,
    dim1: int,
    dim2: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::diagonal_backward.out(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.diagonal_backward.out ge_converter is not implemented!")
