from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(64, 128)),
    Support(BF16(64, 128)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_dynamic_block_quant.default)
def conveter_npu_grouped_dynamic_block_quant_default(
    x: Tensor,
    group_list: Tensor,
    min_scale: float = 0.0,
    round_mode: str = "rint",
    dst_type: int = 291,
    row_block_size: int = 1,
    col_block_size: int = 128,
    group_list_type: int = 0,
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_grouped_dynamic_block_quant(Tensor x, 
                                              Tensor group_list,  
                                              *,
                                              float min_scale=0.0, 
                                              str round_mode="rint",
                                              int dst_type=291, 
                                              int row_block_size=1,
                                              int col_block_size=128,
                                              int group_list_type=0) -> (Tensor y, Tensor scale)
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale = ge.GroupedDynamicBlockQuant(x, group_list, min_scale=min_scale, round_mode=round_mode, dst_type=acl_dst_type,
                                    row_block_size=row_block_size, col_block_size=col_block_size, group_list_type=group_list_type)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    return y, scale