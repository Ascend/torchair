import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import I8, Support


@declare_supported([
    Support(I8(1024))
])

@register_fx_node_ge_converter(torch.ops.npu.attention_worker_scheduler.default)
def conveter_npu_attention_worker_scheduler(
    schedule_context: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::attention_worker_scheduler(Tensor schedule_context) -> Tensor
    """
    copy = ge.TensorMove(schedule_context)
    return ge.AttentionWorkerScheduler(copy)