from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(3, 4), I32(3, 2), I32(3, 2), active_num=3),
    ]
)

@register_fx_node_ge_converter(torch.ops.npu.npu_moe_init_routing.default)
def conveter_npu_moe_init_routing_default(
    x: Tensor,
    row_idx: Tensor,
    expert_idx: Tensor,
    active_num: int = 99,
    meta_outputs: TensorSpec = None,
):
    """NB: func: npu_moe_init_routing(Tensor x, Tensor row_idx, Tensor expert_idx, int active_num)
    -> (Tensor, Tensor, Tensor)
    """
    return ge.MoeInitRouting(x, row_idx, expert_idx, active_num=active_num)
