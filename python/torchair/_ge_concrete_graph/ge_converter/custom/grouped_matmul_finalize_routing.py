from typing import Optional
from typing import List
import torch
from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_proto_type, torch_dtype_value_to_ge_type


def convert_tensorlist_to_mxfp4_item(input_data: Tensor, x_dtype, trans=False):
    shape_multiples = 2
    x_ge_dtype = 0
        
    if x_dtype is not None:
        x_ge_dtype = torch_dtype_value_to_ge_type(x_dtype)
    const_x = ge.Const([1] * (input_data.rank - 1) + [shape_multiples])
    perm = [i for i in range(input_data.rank)]
    perm[-1], perm[-2] = perm[-2], perm[-1]
    if trans:
        input_data = ge.Transpose(input_data, perm)
    shape_x = ge.Shape(input_data)
    shape_x = ge.Mul(shape_x, const_x)
    input_data = ge.Bitcast(input_data, type=x_ge_dtype)
    input_data = ge.Reshape(input_data, shape_x)
    if trans:
        input_data = ge.Transpose(input_data, perm)
    return input_data


@declare_supported([
])
@register_fx_node_ge_converter(torch.ops.npu.npu_grouped_matmul_finalize_routing.default)
def conveter_npu_grouped_matmul_finalize_routing(
    x: Tensor,
    w: Tensor,
    group_list: Tensor,
    *,
    scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    shared_input: Optional[Tensor] = None, 
    logit: Optional[Tensor] = None,
    row_index: Optional[Tensor] = None,
    dtype: torch.dtype = None,
    shared_input_weight: float = 1.0,
    shared_input_offset: int = 0,
    output_bs: int = 0,
    group_list_type: int = 1,
    tuning_config: Optional[List[int]] = None,
    x_dtype: Optional[int] = None,
    w_dtype: Optional[int] = None,
    scale_dtype: Optional[int] = None,
    pertoken_scale_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_grouped_matmul_finalize_routing(Tensor x, Tensor w, Tensor group_list, *,
                        Tensor? scale=None, Tensor? bias=None, Tensor? offset=None,
                        Tensor? pertoken_scale=None, Tensor? shared_input=None,
                        Tensor? logit=None, Tensor? row_index=None, ScalarType? dtype=None,
                        float? shared_input_weight=1.0, int? shared_input_offset=0,
                        int? output_bs=0, int? group_list_type=1, int[] tuning_config=[]) -> Tensor
    """
    try:
        import torch_npu
    except ImportError as e:
        raise RuntimeError(
            "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
            "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e
    tuning_config = tuning_config or [0]
    if dtype is None or dtype == torch.float32:
        dtype = DataType.DT_FLOAT
    else:
        raise RuntimeError("Not supported output dtype is " + str(dtype))
    if_mxfp4 = (x_dtype == torch_npu.float4_e2m1fn_x2 or x_dtype == torch_npu.float4_e1m2fn_x2) and \
        (w_dtype == torch_npu.float4_e2m1fn_x2 or w_dtype == torch_npu.float4_e1m2fn_x2)
    if w.dtype == DataType.DT_INT32:
        from torch_npu.npu.utils import _is_gte_cann_version
        if _is_gte_cann_version("8.5.0"):
            new_w = ge.Bitcast(w, type=DataType.DT_INT4, keep_dim=True)
        else:
            const_w = ge.Const([1] * (w.rank - 1) + [8])
            shape_w = ge.Shape(w)
            shape_w = ge.Mul(shape_w, const_w)
            new_w = ge.Bitcast(w, type=DataType.DT_INT4)
            new_w = ge.Reshape(new_w, shape_w)
    elif x_dtype is not None and w_dtype is not None and if_mxfp4:
        x = convert_tensorlist_to_mxfp4_item(x, x_dtype)
        new_w = convert_tensorlist_to_mxfp4_item(w, w_dtype)
    else:
        new_w = w

    if x_dtype is not None:
        if x_dtype != torch_npu.float4_e2m1fn_x2 and x_dtype != torch_npu.float4_e1m2fn_x2:
            x = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
    if w_dtype is not None:
        if w_dtype != torch_npu.float4_e2m1fn_x2 and w_dtype != torch_npu.float4_e1m2fn_x2:
            new_w = ge.Bitcast(w, type=torch_dtype_value_to_ge_type(w_dtype))
        new_w.desc.dtype = torch_dtype_value_to_ge_proto_type(w_dtype)
    if scale_dtype is not None:
        scale = ge.Bitcast(scale, type=torch_dtype_value_to_ge_type(scale_dtype))
        scale.desc.dtype = torch_dtype_value_to_ge_proto_type(scale_dtype)
    if pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype))
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(pertoken_scale_dtype)

    if output_bs == 0:
        output_bs = x.symsize[0]
    return ge.GroupedMatmulFinalizeRouting(x,
                                 new_w,
                                 scale=scale,
                                 bias=bias,
                                 pertoken_scale=pertoken_scale,
                                 group_list=group_list,
                                 shared_input=shared_input,
                                 logit=logit,
                                 row_index=row_index,
                                 offset=offset,
                                 dtype=dtype,
                                 shared_input_weight=shared_input_weight,
                                 shared_input_offset=shared_input_offset,
                                 transpose_x=False,
                                 transpose_w=False,
                                 output_bs=output_bs,
                                 group_list_type=group_list_type,
                                 tuning_config=tuning_config)
