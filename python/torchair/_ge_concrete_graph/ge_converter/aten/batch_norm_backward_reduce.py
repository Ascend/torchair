from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(32, 32, 112, 112), F32(32, 32, 112, 112), F32(32), F32(32), F32(32), True, True, True),
])
@register_fx_node_ge_converter(torch.ops.aten.batch_norm_backward_reduce.default)
def conveter_aten_batch_norm_backward_reduce_default(
    grad_out: Tensor, 
    self: Tensor, 
    mean: Tensor,
    invstd: Tensor, 
    weight: Optional[Tensor],
    input_g: bool, 
    weight_g: bool, 
    bias_g: bool,
    meta_outputs: TensorSpec = None
):
    if grad_out.rank != 4:
        raise NotImplementedError("torch.ops.aten.batch_norm_backward_reduce.default ", \
            "ge_converter is only implemented for 4D input!")
    if not (input_g and weight_g and bias_g):
        raise NotImplementedError("torch.ops.aten.batch_norm_backward_reduce.default ", \
            "ge_converter is implemented while input, weight and bias all have gradients.")
    axes = list(range(self.rank))
    del axes[1]

    mul_dy_dx = ge.Mul(grad_out, self)
    sum_dy_xmu_sum = ge.ReduceSum(mul_dy_dx, axes, keep_dims=False)
    sum_dy_sum = ge.ReduceSum(grad_out, axes, keep_dims=False) 
    sum_dy_xmu_out, grad_weight_res = ge.SyncBatchNormBackwardReduce(sum_dy_sum, sum_dy_xmu_sum, mean, invstd)
    specific_op_input_layout(sum_dy_xmu_out, indices=list(range(4)), layout="NCHW")
    specific_op_output_layout(sum_dy_xmu_out, indices=list(range(2)), layout="NCHW")
    ret = (sum_dy_sum, sum_dy_xmu_out, grad_weight_res, sum_dy_sum)
    return ret
