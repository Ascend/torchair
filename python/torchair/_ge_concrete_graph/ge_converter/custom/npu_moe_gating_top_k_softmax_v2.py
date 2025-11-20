from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(x = F32(3, 4), k = 1, finished = BOOL(3), renorm = 0, output_softmax = False),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_moe_gating_top_k_softmax_v2.default)
def conveter_npu_moe_gating_top_k_softmax_default(
        x: Tensor,
        *,
        k: int = 1,
        finished: Optional[Tensor] = None,
        renorm: Optional[int] = 0,
        output_softmax: Optional[bool] = False,
        meta_outputs: TensorSpec = None,
):
    """- func: npu_moe_gating_top_k_softmax_v2(Tensor x, *, int k=1, Tensor? finished=None, 
    int? renorm=0, bool? output_softmax=False) -> (Tensor, Tensor, Tensor)
    """
    return ge.MoeGatingTopKSoftmax(
        x = x, k = k, finished = finished, renorm = renorm, output_softmax = output_softmax)