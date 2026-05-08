from typing import List, Optional

import torch
import torchair

from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_chunk_gated_delta_rule.default)
def convert_npu_chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    beta: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    actual_seq_lengths: Optional[Tensor] = None,
    scale: Optional[float] = None,
    g: Optional[Tensor] = None,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: npu::npu_chunk_gated_delta_rule(Tensor query, Tensor key, Tensor value, *, Tensor? beta=None,
    Tensor? initial_state=None, Tensor? actual_seq_lengths=None, float? scale=None, Tensor? g=None)
    -> (Tensor, Tensor)
    """

    return torchair.ge.custom_op(
        "ChunkGatedDeltaRule",

        inputs={
            "query": query,
            "key": key,
            "value": value,
            "beta": beta,
            "initial_state": initial_state,
            "actual_seq_lengths": actual_seq_lengths,
            "g": g,
        },

        attrs={
            "scale_value": attr.Float(scale)
        },

        outputs=["out", "final_state"]
    )