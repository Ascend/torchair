from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 1, 5), input_sizes=(4, 3, 1, 5), dim=1, index=2),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.select_backward.default)
def conveter_aten_select_backward_default(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    index: Union[int, Tensor],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor"""
    
    size_ = ge.Cast(input_sizes, dst_type=torch_type_to_ge_type(torch.int))
    grad_input_ = ge.Empty(size_, dtype=grad_output.dtype)
    
    index_ = ge.BroadcastTo(index, input_sizes)
    grad_output_ = ge.BroadcastTo(ge.ExpandDims(grad_output, dim), input_sizes)

    return ge.ScatterElements(grad_input_, index_, grad_output_, axis=dim)


@register_fx_node_ge_converter(torch.ops.aten.select_backward.out)
def conveter_aten_select_backward_out(
    grad_output: Tensor,
    input_sizes: Union[List[int], Tensor],
    dim: int,
    index: Union[int, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::select_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError(
        "torch.ops.aten.select_backward.out is redundant before pytorch 2.1.0,might be supported in future version.")
