from torchair._ge_concrete_graph.ge_converter.converter_utils import *

COMM_MODE_SUPPOET_LIST = ["ccu", "ai_cpu", ""]


@register_fx_node_ge_converter(torch.ops.npu.npu_alltoallv_gmm.default)
def convert_npu_alltoallv_gmm(
    gmm_x: Tensor,
    gmm_weight: Tensor,
    hcom: str,
    ep_world_size: int,
    send_counts: List[int],
    recv_counts: List[int],
    *,
    send_counts_tensor: Optional[Tensor] = None,
    recv_counts_tensor: Optional[Tensor] = None,
    mm_x: Optional[Tensor] = None,
    mm_weight: Optional[Tensor] = None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    permute_out_flag: bool = False,
    comm_mode: str = None,
    meta_outputs: TensorSpec = None
):
    if comm_mode is None:
        pass
    elif comm_mode not in COMM_MODE_SUPPOET_LIST:
        raise RuntimeError(f"The comm_mode only supports value in {COMM_MODE_SUPPOET_LIST}, but got {comm_mode}.")
    return ge.AlltoAllvGroupedMatMul(
        gmm_x=gmm_x,
        gmm_weight=gmm_weight,
        send_counts_tensor=send_counts_tensor,
        recv_counts_tensor=recv_counts_tensor,
        mm_x=mm_x,
        mm_weight=mm_weight,
        group=hcom,
        ep_world_size=ep_world_size,
        send_counts=send_counts,
        recv_counts=recv_counts,
        trans_gmm_weight=trans_gmm_weight,
        trans_mm_weight=trans_mm_weight,
        permute_out_flag=permute_out_flag,
        comm_mode=comm_mode)
