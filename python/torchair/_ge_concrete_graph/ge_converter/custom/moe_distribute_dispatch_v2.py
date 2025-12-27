from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type


@register_fx_node_ge_converter(torch.ops.npu.npu_moe_distribute_dispatch_v2.default)
def convert_npu_moe_distribute_dispatch_v2(
    x: Tensor,
    expert_ids: Tensor,
    group_ep: str,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    *,
    scales: Optional[Tensor] = None,
    x_active_mask: Optional[Tensor] = None,
    expert_scales: Optional[Tensor] = None,
    elastic_info: Optional[Tensor] = None,
    performance_info: Optional[Tensor] = None,
    group_tp: str = "",
    tp_world_size: int = 0,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 1,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    comm_alg: str = "",
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
    y_dtype: Optional[int] = None,
    x_dtype: Optional[int] = None,
    scales_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    class DispatchResults(NamedTuple):
        expand_x: Tensor
        dynamic_scales: Tensor
        expand_idx: Tensor
        expert_token_nums: Tensor
        ep_recv_count: Tensor
        tp_recv_count: Tensor
        expand_scales: Tensor

    if x_dtype is not None:
        x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if scales_dtype is not None:
        scales = ge.Bitcast(scales, type=torch_dtype_value_to_ge_type(scales_dtype))
        scales.desc.dtype = torch_dtype_value_to_ge_proto_type(scales_dtype)

    expand_x_dtype = DataType.DT_INT8
    if quant_mode == 0:
        expand_x_dtype = x.dtype
    elif y_dtype is not None:
        expand_x_dtype = torch_dtype_value_to_ge_type(y_dtype)

    (expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_count, tp_recv_count, expand_scales) = \
        ge.MoeDistributeDispatchV2(x,
                                   expert_ids,
                                   scales=scales,
                                   x_active_mask=x_active_mask,
                                   expert_scales=expert_scales,
                                   elastic_info=elastic_info,
                                   performance_info=performance_info,
                                   group_ep=group_ep,
                                   ep_world_size=ep_world_size,
                                   ep_rank_id=ep_rank_id,
                                   moe_expert_num=moe_expert_num,
                                   group_tp=group_tp,
                                   tp_world_size=tp_world_size,
                                   tp_rank_id=tp_rank_id,
                                   expert_shard_type=expert_shard_type,
                                   shared_expert_num=shared_expert_num,
                                   shared_expert_rank_num=shared_expert_rank_num,
                                   quant_mode=quant_mode,
                                   global_bs=global_bs,
                                   expert_token_nums_type=expert_token_nums_type,
                                   comm_alg=comm_alg,
                                   zero_expert_num=zero_expert_num,
                                   copy_expert_num=copy_expert_num,
                                   const_expert_num=const_expert_num,
                                   y_dtype=expand_x_dtype)

    expand_x.desc.dtype = ge_dtype_to_ge_proto_dtype(expand_x_dtype)

    dynamic_scales_dtype = DataType.DT_FLOAT
    if quant_mode == 0:
        if x.dtype not in (DataType.DT_FLOAT16, DataType.DT_BF16) and scales is not None:
            dynamic_scales_dtype = scales.dtype
    elif quant_mode == 4:
        dynamic_scales_dtype = DataType.DT_FLOAT8_E8M0
    dynamic_scales.desc.dtype = ge_dtype_to_ge_proto_dtype(dynamic_scales_dtype)
    dispatch_results = DispatchResults(expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_count, tp_recv_count, expand_scales)
    return dispatch_results