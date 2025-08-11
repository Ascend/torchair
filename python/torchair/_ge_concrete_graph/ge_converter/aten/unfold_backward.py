from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.unfold_backward.default)
def conveter_aten_unfold_backward_default(
    grad_in: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    size: int,
    step: int,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.unfold_backward.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.unfold_backward.out)
def conveter_aten_unfold_backward_out(
    grad_in: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    size: int,
    step: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::unfold_backward.out(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.unfold_backward.out ge_converter is not implemented!")
