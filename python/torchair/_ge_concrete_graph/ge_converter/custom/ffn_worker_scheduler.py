import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair._ge_concrete_graph.supported_declaration import I8, Support


@declare_supported([
    Support(I8(1024),
            sync_group_size=1,
            execute_mode=0)
])

@register_fx_node_ge_converter(torch.ops.npu.ffn_worker_scheduler.default)
def conveter_npu_ffn_worker_scheduler(
    schedule_context: Tensor,
    *,
    sync_group_size: int = 1,
    execute_mode: int = 0,

    meta_outputs: TensorSpec = None,
):
    """NB: npu::ffn_worker_scheduler(Tensor schedule_context, *, sync_group_size=1,
    execute_mode=0) -> Tensor
    """
    copy = ge.TensorMove(schedule_context)
    return ge.FfnWorkerScheduler(
        copy,
        sync_group_size=sync_group_size, execute_mode=execute_mode
    )