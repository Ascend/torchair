from typing import List, Optional

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import DataType, Tensor, TensorSpec, torch_dtype_value_to_ge_proto_type, \
    torch_dtype_value_to_ge_type

@register_fx_node_ge_converter(torch.ops.npu_inference.npu_fused_quant_matmul.default)
def converter_npu_npu_fused_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    x3: Optional[Tensor] = None,
    fused_op_type: Optional[str] = "",
    output_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    pertoken_scale_dtype: int = None,
    scale_dtype: int = None,
    x3_dtype: int = None,
    group_sizes: Optional[List[int]] = None,
    y_scale: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_fused_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None,
            Tensor? pertoken_scale=None, Tensor? bias=None, Tensor? x3=None, str fused_op_type='',
            int? output_dtype=None, int? x1_dtype=None, int? x2_dtype=None, int? pertoken_scale_dtype=None,
            int? scale_dtype=None, int? x3_dtype=None, int[]? group_sizes=None, Tensor? y_scale=None) -> Tensor
    """
    import torch_npu

    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)

    if x2_dtype is not None and x2_dtype == torch_npu.int4:
        x2 = ge.Bitcast(x2, type=DataType.DT_INT4, keep_dim=True)

    if scale is not None and scale.dtype == DataType.DT_INT64:
        scale = ge.Bitcast(scale, type=DataType.DT_UINT64)
    group_size = 0
    out = ge.FusedQuantMatMul(x1,
                              x2,
                              bias=bias,
                              x1_scale=pertoken_scale,
                              x2_scale=scale,
                              y_scale=y_scale,
                              x1_offset=None,
                              x2_offset=offset,
                              y_offset=None,
                              x2_table=None,
                              x3=x3,
                              dtype=dtype,
                              transpose_x1=False,
                              transpose_x2=False,
                              group_size=group_size,
                              fused_op_type=fused_op_type)
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out
