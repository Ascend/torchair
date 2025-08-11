from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def squeeze_and_expand(inp, dst_shape):
    inp = ge.Unsqueeze(inp, axes=[0])
    inp = ge.Unsqueeze(inp, axes=list(range(2, 4)))
    inp = ge.BroadcastTo(inp, dst_shape)
    return inp


@declare_supported([
    Support(F32(32, 32, 112, 112), F32(32, 32, 112, 112), F32(32), F32(32), F32(32), F32(32), F32(32), F32(8, 1)),
])
@register_fx_node_ge_converter(torch.ops.aten.batch_norm_backward_elemt.default)
def conveter_aten_batch_norm_backward_elemt_default(
    grad_out: Tensor,
    inp: Tensor,
    mean: Tensor,
    invstd: Tensor,
    weight: Optional[Tensor],
    sum_dy: Tensor,
    sum_dy_xmu: Tensor,
    count: Tensor,
    meta_outputs: TensorSpec = None
):
    if grad_out.rank != 4:
        raise NotImplementedError("torch.ops.aten.batch_norm_backward_elemt.default ", \
            "ge_converter is only implemented for 4D input!")
    axes = list(range(count.rank))
    divisor = ge.ReduceSum(count, axes, keep_dims=False)
    mean_dy = ge.RealDiv(sum_dy, divisor)
    mean_dy_xmu = ge.RealDiv(sum_dy_xmu, divisor)

    dst_shape = ge.Shape(inp)
    mean_expanded = squeeze_and_expand(mean, dst_shape)
    invstd_expanded = squeeze_and_expand(invstd, dst_shape)
    weight_expanded = squeeze_and_expand(weight, dst_shape)
    mean_dy_expanded = squeeze_and_expand(mean_dy, dst_shape)
    mean_dy_xmu_expanded = squeeze_and_expand(mean_dy_xmu, dst_shape)
    grad_input = ge.SyncBatchNormBackwardElemt(grad_out, inp, mean_expanded, invstd_expanded, \
        weight_expanded, mean_dy_expanded, mean_dy_xmu_expanded)
    specific_op_input_layout(grad_input, indices=list(range(7)), layout="NCHW")
    specific_op_output_layout(grad_input, indices=0, layout="NCHW")
    return grad_input
