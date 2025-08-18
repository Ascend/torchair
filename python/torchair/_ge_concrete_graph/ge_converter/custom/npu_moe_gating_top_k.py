from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_gating_top_k.default)
def conveter_npu_moe_gating_top_k_default(
        x: Tensor,
        k: int,
        *,
        bias: Optional[Tensor] = None,
        k_group: int = 1,
        group_count: int = 1,
        group_select_mode: int = 0,
        renorm: int = 0,
        norm_type: int = 0,
        out_flag: bool = False,
        routed_scaling_factor: float = 1.0,
        eps: float = 1e-20,
        meta_outputs: List[TensorSpec] = None,
):
    return ge.MoeGatingTopK(x, bias, k=k, k_group=k_group, group_count=group_count, 
                            group_select_mode=group_select_mode, renorm=renorm, norm_type=norm_type, 
                            out_flag=out_flag, routed_scaling_factor=routed_scaling_factor, eps=eps)