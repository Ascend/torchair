from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(8, 256), F32(1)),
    Support(BF16(4, 16, 128), F32(1)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_quant_max.default)
def conveter_npu_quant_max_default(
    x: Tensor,
    scale: Tensor,
    round_mode: str = "rint",
    dst_dtype: int = 291,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_quant_max(Tensor x, Tensor scale, *,
                            str round_mode="rint",
                            int dst_dtype=torch_npu.float8_e5m2) -> (Tensor y, Tensor amax)
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_dtype)
    y, amax = ge.QuantMax(x, scale, round_mode=round_mode, dst_type=acl_dst_type)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_dtype)
    return y, amax
