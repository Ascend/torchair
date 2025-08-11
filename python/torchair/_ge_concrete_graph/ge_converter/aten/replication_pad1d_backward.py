from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.replication_pad1d_backward.default)
def conveter_aten_replication_pad1d_backward_default(
        grad_output: Tensor,
        self: Tensor,
        padding: Union[List[int], Tensor],
        meta_outputs: TensorSpec = None,
):
    """NB: aten::replication_pad1d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor"""
    if isinstance(padding, Tensor):
        raise NotImplementedError(
            "When padding is Tensor, torch.ops.aten.replication_pad1d.default ge_converter is not implemented!")
    if len(padding) < 2:
        raise AssertionError("padding length shoud be at least 2")

    grad_output_cp = ge.Unsqueeze(grad_output, axes=[0])
    if sum(padding) == 0:
        return ge.Identity(grad_output)
    padding = [0, 0, 0, 0, 0, 0, padding[0], padding[1]]
    grad_input = ge.PadV3Grad(grad_output_cp, padding, mode="edge", paddings_contiguous=True)
    return ge.Squeeze(grad_input, axis=[0])


@register_fx_node_ge_converter(torch.ops.aten.replication_pad1d_backward.grad_input)
def conveter_aten_replication_pad1d_backward_grad_input(
        grad_output: Tensor,
        self: Tensor,
        padding: Union[List[int], Tensor],
        *,
        grad_input: Tensor = None,
        meta_outputs: TensorSpec = None
):
    """NB: aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)"""
    raise AssertionError("torch.ops.aten.replication_pad1d_backward.out is redundant before pytorch 2.1.0,"
                         " might be supported in future version.")
