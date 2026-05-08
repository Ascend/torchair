from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import torch_dtype_value_to_ge_proto_type


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_re_routing.default)
def conveter_npu_moe_re_routing_default(
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        *,
        per_token_scales: Optional[Tensor] = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
        tokens_dtype: Optional[int] = None,
        meta_outputs: List[TensorSpec] = None):
    if tokens_dtype is not None:
        tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)
    
    permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num = \
        ge.MoeReRouting(tokens, expert_token_num_per_rank, per_token_scales,
                        expert_token_num_type=expert_token_num_type,
                        idx_type=idx_type)
    
    if tokens_dtype is not None:
        permute_tokens.desc.dtype = torch_dtype_value_to_ge_proto_type(tokens_dtype)
    
    return permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num