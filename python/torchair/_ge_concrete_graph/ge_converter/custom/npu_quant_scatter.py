from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import dont_prune_me


ROUND_MODE_SUPPORT_MAP = {
    DataType.DT_HIFLOAT8: ["round", "hybrid"],
    DataType.DT_FLOAT8_E4M3FN: ["rint"],
    DataType.DT_FLOAT8_E5M2: ["rint"],
    DataType.DT_INT8: ["rint"],
}


@declare_supported([
    Support(I8(24, 4096, 128), I32(24), BF16(24, 1, 128), BF16(1, 1, 128), BF16(1, 1, 128),
            axis=-2, quant_axis=-1, reduce="update", dst_type=1, round_mode="rint"),
    Support(I8(24, 4096, 128), I32(24), BF16(24, 1, 128), BF16(1, 1, 128),
            axis=-2, quant_axis=-1, reduce="update", dst_type=1, round_mode="rint"),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_quant_scatter.default)
def converter_npu_quant_scatter_default(
        self: Tensor,
        indices: Tensor,
        updates: Tensor,
        quant_scales: Tensor,
        quant_zero_points: Optional[Tensor] = None,
        axis: int = -2,
        quant_axis: int = -1,
        reduce: str = 'update',
        dst_type: Optional[int] = None,
        round_mode: Optional[str] = "rint",
        meta_outputs: TensorSpec = None
):
    """
    NB: aten::npu_quant_scatter(Tensor self, Tensor indices, Tensor updates, Tensor quant_scales,
                                Tensor? quant_zero_points=None, int axis=0, int quant_axis=1,
                                str reduce='update', dst_type=1, round_mode="rint" ) -> Tensor
    """
    """
    Warning: kernel [npu_quant_scatter] is a out-of-place op, but it is supported by another in-place op 
             cann.QuantUpdateScatter. This current usage may cause the input to be changed unexpectedly, and the caller 
             needs to pay attention to this feature.
    """
    import torch_npu
    if dst_type is not None:
        dst_ge_dtype = torch_dtype_value_to_ge_type(dst_type)
        dst_ge_proto_dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    else:
        dst_ge_dtype = DataType.DT_INT8
        dst_ge_proto_dtype = ProtoDataType.DT_INT8
    if dst_ge_dtype not in [DataType.DT_INT8, DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2, DataType.DT_HIFLOAT8]:
        raise RuntimeError(f"dst_type:{dst_type} not supported, only supports torch.int8(1), "
                            "torch.float8_e5m2(23), torch.float8_e4m3fn(24), torch_npu.hifloat8(290)")

    self.desc.dtype = dst_ge_proto_dtype
    if dst_type == torch_npu.hifloat8:
        self = ge.Bitcast(self, type=DataType.DT_HIFLOAT8)
    copy = ge.TensorMove(self)
    copy.desc.dtype = dst_ge_proto_dtype

    round_mode_support_list = ROUND_MODE_SUPPORT_MAP.get(dst_ge_dtype)
    if round_mode is None:
        round_mode = "rint"
    if round_mode not in round_mode_support_list:
        raise RuntimeError(f"round_mode not supported for dst_type:{dst_type}, support lists are:{round_mode_support_list}")
    round_mode_str = round_mode
    y = ge.QuantUpdateScatter(copy, indices, updates, quant_scales, quant_zero_points, reduce=reduce, axis=axis,
                              quant_axis=quant_axis, round_mode=round_mode_str)
    y.desc.dtype = dst_ge_proto_dtype
    return y


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_scatter_.default)
def conveter_npu_quant_scatter__default(
        self: Tensor,
        indices: Tensor,
        updates: Tensor,
        quant_scales: Tensor,
        quant_zero_points: Optional[Tensor] = None,
        axis: int = 0,
        quant_axis: int = 1,
        reduce: str = 'update',
        meta_outputs: TensorSpec = None
):
    """
    NB: func: npu_quant_scatter_(Tensor(a!) self, Tensor indices, Tensor updates, Tensor quant_scales,
                                 Tensor? quant_zero_points=None, int axis=0, int quant_axis=1,
                                 str reduce='update') -> Tensor(a!)
    """

    """
    The converter for inplace operators is generally not necessary, 
    because all inplace operators become non_inplace operators after functionalization.
    Adding converters to those inplace operators is due to the implementation of some re-inplace pass, 
    which pass can transfer some non_inplace operators to the original inplace operators.
    """

    op = ge.QuantUpdateScatter(self, indices, updates, quant_scales, quant_zero_points, reduce=reduce, axis=axis,
                               quant_axis=quant_axis)
    dont_prune_me(op)
    return op
