from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import torch_dtype_value_to_ge_type, torch_dtype_value_to_ge_proto_type


@declare_supported([
    Support(F16(2, 8))
])
@register_fx_node_ge_converter(torch.ops.npu.npu_swiglu_mx_quant.default)
def converter_npu_swiglu_mx_quant_default(
        x: Tensor,
        group_index: Tensor = None,
        activate_left: bool = False,
        activate_dim: int = -1,
        swiglu_mode: int = 0,
        clamp_limit: float = 7.0,
        glu_alpha: float = 1.702,
        glu_bias: float = 1.0,
        group_mode: int = 0,
        axis: int = -1,
        dst_type: int = 296,
        round_mode: str = "rint",
        scale_alg: int = 0,
        max_dtype_value: float = 0,
        meta_outputs: List[TensorSpec] = None):
    """
    NB: aten::npu_swiglu_mx_quant(Tensor x, *, Tensor? group_index=None,
                                   bool activate_left=False, int activate_dim=-1,
                                   int swiglu_mode=0, float clamp_limit=7.0,
                                   float glu_alpha=1.702, float glu_bias=1.0,
                                   int group_mode=0, int axis=-1, int dst_type=296,
                                   str round_mode="rint", int scale_alg=0,
                                   float max_dtype_value=0) -> (Tensor y, Tensor scale)
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, scale = ge.SwigluMxQuant(x, group_index=group_index,
                                activate_left=activate_left, activate_dim=activate_dim,
                                swiglu_mode=swiglu_mode, clamp_limit=clamp_limit,
                                glu_alpha=glu_alpha, glu_bias=glu_bias,
                                group_mode=group_mode, axis=axis, dst_type=acl_dst_type,
                                round_mode=round_mode, scale_alg=scale_alg,
                                max_dtype_value=max_dtype_value)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    scale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0

    # torch_npu.float4_e2m1fn_x2:296,torch_npu.float4_e1m2fn_x2:297
    if dst_type == 296 or dst_type == 297:
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(2)
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_uint8 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_2bit = ge.ConcatV2([y_shape_uint8, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_2bit), type=DataType.DT_UINT8)
        return ge.Reshape(y, y_shape_uint8), scale
    else:
        return y, scale
