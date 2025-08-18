from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(1), offset=None)
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_trans_quant_param.default)
def conveter_npu_npu_trans_quant_param(
    scale: Tensor,
    offset: Optional[Tensor] = None,
    round_mode: Optional[int] = 0,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_trans_quant_param(Tensor scale, Tensor? offset=None, int? round_mode=0) -> Tensor"""
    return ge.TransQuantParamV2(scale, offset=offset, round_mode=round_mode)
