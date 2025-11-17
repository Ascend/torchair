from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(32, 32, 112, 112), F32(32), F32(32), F32(32), F32(32), 1e-7),
])
@register_fx_node_ge_converter(torch.ops.aten.batch_norm_elemt.default)
def conveter_aten_batch_norm_elemt_default(
    inp: Tensor, 
    weight: Optional[Tensor], 
    bias: Optional[Tensor], 
    running_mean: Tensor, 
    running_var: Tensor, 
    eps: float, 
    meta_outputs: TensorSpec = None
):
    if inp.rank != 4:
        raise NotImplementedError("torch.ops.aten.batch_norm_elemt.default ", \
            "ge_converter is only implemented for 4D input!")
    var = ge.Mul(running_var, running_var)
    var = ge.Div(1.0, var)
    var = ge.Sub(var, eps)

    if is_arch35():
        output, _, _, _, _ = ge.BatchNormV3(inp, weight, bias, running_mean, var,
                                            epsilon=eps, momentum=0.10, is_training=False)
    else:
        output = ge.BNInfer(inp, weight, bias, running_mean, var, epsilon=eps)
    specific_op_input_layout(output, indices=list(range(5)), layout="NCHW")
    specific_op_output_layout(output, indices=0, layout="NCHW")
    return output
