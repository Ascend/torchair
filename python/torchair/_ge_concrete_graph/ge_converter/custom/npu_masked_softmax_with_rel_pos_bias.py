from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported( # 典型shape
    [
        Support(F16(96, 2, 2, 32, 32), F16(2, 32, 32), F16(2, 32, 32),
            scale_value=1, inner_precision_mode=0),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_masked_softmax_with_rel_pos_bias.default)
def convert_npu_npu_masked_softmax_with_rel_pos_bias(
    x: Tensor,
    atten_mask: Optional[Tensor],
    relative_pos_bias: Tensor,
    scale_value: float = 1.0,
    inner_precision_mode: int = 0,
    meta_outputs: TensorSpec = None
):
    return ge.MaskedSoftmaxWithRelPosBias(x, atten_mask, relative_pos_bias, scale_value=scale_value,
        inner_precision_mode=inner_precision_mode)
