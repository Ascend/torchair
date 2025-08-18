from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_gmm_alltoallv.default)
def convert_npu_gmm_alltoallv(
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
    meta_outputs: TensorSpec = None
):
    return ge.GroupedMatMulAlltoAllv(
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
        trans_mm_weight=trans_mm_weight)