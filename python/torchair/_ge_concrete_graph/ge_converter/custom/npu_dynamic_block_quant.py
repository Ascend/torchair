from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def torch_dtype_value_to_ge_type(torch_type: int) -> int:
    if torch_type == 23:
        return DataType.DT_FLOAT8_E5M2
    if torch_type == 24:
        return DataType.DT_FLOAT8_E4M3FN

    raise RuntimeError("Unsupported dst_type, only support torch.float8_e5m2,torch.float8_e4m3fn,"
                       "torch_npu.float8_e5m2,torch_npu.float8_e4m3fn,torch_npu.hifloat8.")


@declare_supported([
    Support(F16(64, 128)),
    Support(BF16(64, 128)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_block_quant.default)
def conveter_npu_dynamic_block_quant_default(
    x: Tensor,
    min_scale: float = 0.0,
    round_mode: str = "rint",
    dst_type: int = 1,
    row_block_size: int = 1,
    col_block_size: int = 128,
    meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_dynamic_block_quant(Tensor x, *,
                                      float min_scale=0.0, str round_mode="rint",
                                      int dst_type=1, int row_block_size=1,
                                      int col_block_size=128) -> (Tensor y, Tensor scale)
    """
    if dst_type >= 256:
        acl_dst_type = dst_type - 256
    else:
        acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale = ge.DynamicBlockQuant(x, min_scale=min_scale, round_mode=round_mode, dst_type=acl_dst_type,
                                row_block_size=row_block_size, col_block_size=col_block_size)
    return y, scale