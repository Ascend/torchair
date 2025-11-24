from torchair._ge_concrete_graph.ge_converter.converter_utils import *


ROUND_MODE_SUPPORT_MAP = {
    DataType.DT_HIFLOAT8: ["round", "hybrid"],
    DataType.DT_FLOAT8_E4M3FN: ["rint"],
    DataType.DT_FLOAT8_E5M2: ["rint"],
    DataType.DT_INT8: ["rint"],
}


@register_fx_node_ge_converter(torch.ops.npu.npu_gelu_quant.default)
def conveter_npu_gelu_quant_default(
        self: Tensor,
        *,
        input_scale: Optional[Tensor] = None,
        input_offset: Optional[Tensor] = None,
        approximate: str = "none",
        quant_mode: str = "dynamic",
        dst_type: Optional[int] = None,
        round_mode: str = "rint",
        meta_outputs: TensorSpec = None):
    """ NB: npu::npu_gelu_quant(Tensor self, *, Tensor? input_scale=None, Tensor? input_offset=None, 
                                str approximate='none', str quant_mode='dynamic', int? dst_type=None, 
                                str round_mode='rint') -> (Tensor, Tensor?)"""
    if quant_mode not in ["static", "dynamic"]:
        raise AssertionError(f"quant_mode:{quant_mode} must be 'dynamic' or 'static'.")
    if quant_mode == "static" and input_scale is None:
        raise RuntimeError("If quant_mode is 'static', input_scale must not be None")
    if quant_mode == 'dynamic' and input_scale is None and input_offset is not None:
        raise RuntimeError("If quant_mode is 'dynamic', when input_offset is not None, input_scale must not be None too.")

    dim_num = self.rank
    if 0 <= dim_num < 2 and quant_mode == 'dynamic':
        raise RuntimeError("Tensor self must be at least 2-dimensional when quant_mode is 'dynamic'.")

    if dst_type is not None:
        y_ge_dtype = torch_dtype_value_to_ge_type(dst_type)
    else:
        y_ge_dtype = DataType.DT_INT8
    if y_ge_dtype not in [DataType.DT_INT8, DataType.DT_FLOAT8_E4M3FN, DataType.DT_FLOAT8_E5M2, DataType.DT_HIFLOAT8]:
        raise RuntimeError(f"dst_type:{dst_type} not supported, only supports torch.int8(1), "
                            "torch.float8_e5m2(23), torch.float8_e4m3fn(24), torch_npu.hifloat8(290)")
    round_mode_support_list = ROUND_MODE_SUPPORT_MAP.get(y_ge_dtype)
    if round_mode not in round_mode_support_list:
        raise RuntimeError(f"round_mode not supported for dst_type:{dst_type}, support lists are:{round_mode_support_list}")
    round_mode_str = round_mode
    y, out_scale = ge.GeluQuant(self, input_scale=input_scale, input_offset=input_offset, approximate=approximate, 
                                quant_mode=quant_mode, dst_type=y_ge_dtype, round_mode=round_mode_str)
    if y_ge_dtype == DataType.DT_HIFLOAT8:
        y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    return (y, out_scale)