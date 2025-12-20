from typing import (
    List, Optional
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec


@register_fx_node_ge_converter(torch.ops.npu.npu_ffn_to_attention.default)
def convert_npu_ffn_to_attention(
    x: Tensor,
    session_ids: Tensor,
    micro_batch_ids: Tensor,
    token_ids: Tensor,
    expert_offsets: Tensor,
    actual_token_num: Tensor,
    group: str,
    world_size: int,
    token_info_table_shape: List[int],
    token_data_shape: List[int],
    *,
    attn_rank_table: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):

    return ge.FFNToAttention(x=x,
        session_ids=session_ids,
        micro_batch_ids=micro_batch_ids,
        token_ids=token_ids,
        expert_offsets=expert_offsets,
        actual_token_num=actual_token_num,
        attn_rank_table=attn_rank_table,
        group=group,
        world_size=world_size,
        token_info_table_shape=token_info_table_shape,
        token_data_shape=token_data_shape)