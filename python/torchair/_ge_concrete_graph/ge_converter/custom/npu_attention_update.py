from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_attention_update.default)
def conveter_npu_attention_update_default(
        lse: List[Tensor],
        local_out: List[Tensor],
        update_type: int,
        meta_outputs: TensorSpec = None):

    sp = len(lse)
    return ge.AttentionUpdate(lse, local_out, update_type=update_type, sp=sp)