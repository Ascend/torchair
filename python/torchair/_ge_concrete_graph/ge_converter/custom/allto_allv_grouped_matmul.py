from torchair._ge_concrete_graph.ge_converter.converter_utils import *


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
    meta_outputs: TensorSpec = None
):
    gmm_x_scale = None
    gmm_weight_scale = None
    gmm_x_offset = None
    gmm_weight_offset = None
    mm_x_scale = None
    mm_weight_scale = None
    mm_x_offset = None
    mm_weight_offset = None
    gmm_x_quant_mode = 0
    gmm_weight_quant_mode = 0
    mm_x_quant_mode = 0
    mm_weight_quant_mode = 0
    group_size = 0
    y_dtype = 28
    mm_dtype = 28
    dependencies = []
    node_name = None

    gmm_y, mm_y, permute_out = ge.AlltoAllvGroupedMatMul(
        gmm_x=gmm_x,
        gmm_weight=gmm_weight,
        send_counts_tensor=send_counts_tensor,
        recv_counts_tensor=recv_counts_tensor,
        mm_x=mm_x,
        mm_weight=mm_weight,
        gmm_x_scale=gmm_x_scale,
        gmm_weight_scale=gmm_weight_scale,
        gmm_x_offset=gmm_x_offset,
        gmm_weight_offset=gmm_weight_offset,
        mm_x_scale=mm_x_scale,
        mm_weight_scale=mm_weight_scale,
        mm_x_offset=mm_x_offset,
        mm_weight_offset=mm_weight_offset,
        group=hcom,
        ep_world_size=ep_world_size,
        send_counts=send_counts,
        recv_counts=recv_counts,
        trans_gmm_weight=trans_gmm_weight,
        trans_mm_weight=trans_mm_weight,
        permute_out_flag=permute_out_flag,
        gmm_x_quant_mode=gmm_x_quant_mode,
        gmm_weight_quant_mode=gmm_weight_quant_mode,
        mm_x_quant_mode=mm_x_quant_mode,
        mm_weight_quant_mode=mm_weight_quant_mode,
        group_size=group_size,
        y_dtype=y_dtype,
        mm_dtype=mm_dtype,
        dependencies=dependencies,
        node_name=node_name)
    return gmm_y, mm_y, permute_out