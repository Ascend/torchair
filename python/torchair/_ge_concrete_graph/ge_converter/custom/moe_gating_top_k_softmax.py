from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(3, 4), BOOL(3), k=1),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_moe_gating_top_k_softmax.default)
def conveter_npu_moe_gating_top_k_softmax_default(
        x: Tensor,
        finished: Optional[Tensor] = None,
        k: int = 1,
        meta_outputs: TensorSpec = None,
):
    """NB: func: npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num)
    -> (Tensor, Tensor, Tensor)
    """
    return ge.MoeGatingTopKSoftmax(x, finished, k=k)
