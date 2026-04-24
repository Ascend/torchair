from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F16(128, 2048), F16(256, 256), alpha=0, dst_dtype=16),
    ]
)
@register_fx_node_ge_converter(torch.ops.npu.npu_rotate_quant.default)
def converter_npu_rotate_quant(
    x: Tensor,
    rotation: Tensor,
    *,
    alpha: float = 0.000000,
    dst_dtype: int = 0,
    meta_outputs: TensorSpec = None
):
    """
    NPU旋转量化算子的GE图转换器，用于将torch.ops.npu.npu_rotate_quant.default算子
    转换为GE（Graph Engine）可识别的计算图节点。
    Args:
        x (Tensor): 输入特征张量，通常为待旋转量化的原始激活/特征图
        rotation (Tensor): 旋转角度/位置编码张量，与输入x进行旋转计算
        alpha (float, optional): 预留
        dst_dtype (int, optional): 目标输出数据类型标识
        meta_outputs (TensorSpec, optional): 输出张量的元信息（shape/dtype/device），
            用于静态图推导输出结构，默认为 None

    Returns:
        Any: 构造完成的GE图节点，代表npu_rotate_quant算子在GE中的执行逻辑
    """
    acl_dst_type = torch_dtype_value_to_ge_type(dst_dtype)
    y, scale = ge.RotateQuant(x, rotation, y_dtype=acl_dst_type, alpha=alpha)
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_dtype)
    if dst_dtype == 16: # torch.quint4x2
        dim_num = x.rank
        bit_shape = []
        for _ in range(dim_num - 1):
            bit_shape.append(1)
        bit_shape.append(8)
        div_x2 = ge.Const(bit_shape, dtype=DataType.DT_INT32)
        y_shape_int4 = ge.Shape(y)
        y_shape_int32 = ge.Div(y_shape_int4, div_x2)
        y_shape_int4_4bit = ge.ConcatV2([y_shape_int32, ge.Const([8], dtype=DataType.DT_INT32)],
                                        concat_dim=0, N=2)
        y = ge.Bitcast(ge.Reshape(y, y_shape_int4_4bit), type=DataType.DT_INT32)
        y = ge.Reshape(y, y_shape_int32)
    return y, scale