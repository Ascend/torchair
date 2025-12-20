from typing import (
    List, Optional
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.npu.npu_attention_to_ffn.default)
def convert_npu_attention_to_ffn(
    x: Tensor,
    session_id: Tensor,
    micro_batch_id: Tensor,
    layer_id: Tensor,
    expert_ids: Tensor,
    expert_rank_table: Tensor,
    group: str,
    world_size: int,
    ffn_token_info_table_shape: List[int],
    ffn_token_data_shape: List[int],
    attn_token_info_table_shape: List[int],
    moe_expert_num: int,
    *,
    scales: Tensor = None,
    active_mask: Tensor = None,
    quant_mode: int = 0,
    sync_flag: int = 0,
    ffn_start_rank_id: int = 0,
    meta_outputs: TensorSpec = None
):

    return ge.AttentionToFFN(x=x,
        session_id=session_id,
        micro_batch_id=micro_batch_id,
        layer_id=layer_id,
        expert_ids=expert_ids,
        expert_rank_table=expert_rank_table,
        scales=scales,
        active_mask=active_mask,
        group=group,
        world_size=world_size,
        ffn_token_info_table_shape=ffn_token_info_table_shape,
        ffn_token_data_shape=ffn_token_data_shape,
        attn_token_info_table_shape=attn_token_info_table_shape,
        moe_expert_num=moe_expert_num,
        quant_mode=quant_mode,
        sync_flag=sync_flag,
        ffn_start_rank_id=ffn_start_rank_id)