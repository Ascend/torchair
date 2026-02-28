from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, torch_dtype_value_to_ge_proto_type, _ge_dtype_to_ge_proto_dtype


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_init_routing_v2.default)
def conveter_npu_moe_init_routing_v2_default(
        x: Tensor,
        expert_idx: Tensor,
        *,
        scale: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        active_num: int = -1,
        expert_capacity: int = -1,
        expert_num: int = -1,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 0,
        expert_tokens_num_flag: bool = False,
        quant_mode: int = -1,
        active_expert_range: List[int] = [],
        row_idx_type: int = 0,
        meta_outputs: List[TensorSpec] = None,
):
    expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale = ge.MoeInitRoutingV3(x, expert_idx, scale, offset,
                               active_num=active_num, expert_capacity=expert_capacity,
                               expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                               expert_tokens_num_type=expert_tokens_num_type,
                               expert_tokens_num_flag=expert_tokens_num_flag,
                               quant_mode=quant_mode, active_expert_range=active_expert_range,
                               row_idx_type=row_idx_type)
    if quant_mode in [7, 8]:
        import torch_npu
        expanded_x.desc.dtype = torch_dtype_value_to_ge_proto_type(torch_npu.hifloat8)
    return expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale
