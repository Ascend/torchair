from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
    torch_dtype_value_to_ge_proto_type
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType


@declare_supported([
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3)),
    Support(F16(1, 2, 3), F16(1, 2, 3), F16(3), beta=F16(3))
])
@register_fx_node_ge_converter(torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant.default)
def conveter_npu_add_rms_norm_dynamic_mx_quant_default(
    x1: Tensor,
    x2: Tensor,
    gamma: Tensor,
    beta: Optional[Tensor] = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = 296,
    meta_outputs: List[TensorSpec] = None
):
    """
    NB: aten::npu_add_rms_norm_dynamic_mx_quant(Tensor x1, Tensor x2, Tensor gamma, *, Tensor? beta=None, 
                                                float epsilon=1e-06, int scale_alg=0, str round_mode="rint", 
                                                int dst_type=296) -> (Tensor, Tensor, Tensor, Tensor)
    """
    # 输出dst_type类型Pytorch不支持，转换
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    y, x, mxscale, rstd = ge.AddRmsNormDynamicMxQuant(x1, x2, gamma, beta=beta, epsilon=epsilon, scale_alg=scale_alg, round_mode=round_mode, dst_type=acl_dst_type, output_rstd=False)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    mxscale.desc.dtype = ProtoDataType.DT_FLOAT8_E8M0
    # 当 dst_dtype 是FP4时，需特殊处理，以FP8类型存储
    if dst_type == 296 or dst_type == 297:
        dim_num = x1.rank
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
        return ge.Reshape(y, y_shape_uint8), x, mxscale, rstd
    else:
        return y, x, mxscale, rstd