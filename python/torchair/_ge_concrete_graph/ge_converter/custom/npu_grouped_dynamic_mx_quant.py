from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_dynamic_mx_quant.default)
def conveter_npu_grouped_dynamic_mx_quant_default(
    x: Tensor,
    group_index: Tensor,
    round_mode: str = "rint",
    dst_type: int = 23, # torch.float8_e5m2 enum value is 23
    blocksize: int = 32,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_grouped_dynamic_mx_quant(Tensor x, Tensor group_index, *, 
                                   str round_mode="rint", int dst_type=torch.float8_e5m2,
                                   int blocksize=32) -> (Tensor y, Tensor mxscale)
    """
    ge_dst_type = torch_dtype_value_to_ge_type(dst_type)
    if ge_dst_type not in [DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2]:
        raise RuntimeError("Parameter dtype only supports torch.float8_e5m2, torch.float8_e4m3fn"
                           "got " + str(dst_type))
    if x.rank != 2:
        raise RuntimeError(f"Input x must be 2-dimensional, got {x.rank}")
    if group_index.rank != 1:
        raise RuntimeError(f"Input group_index must be 1-dimensional, got {group_index.rank}")
    if round_mode != "rint":
        raise ValueError("Parameter round_mode must be 'rint', got " + round_mode)
    if blocksize != 32:
        raise ValueError(f"Parameter blocksize must be 32, got {blocksize}")
    y, mxscale = ge.GroupedDynamicMxQuant(
        x, group_index, round_mode=round_mode, dst_type=ge_dst_type, blocksize=blocksize)
    mxscale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    return y, mxscale