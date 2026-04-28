from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair.ge._ge_graph import DataType, torch_dtype_value_to_ge_type


@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)
def converter_npu_npu_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    output_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    pertoken_scale_dtype: int = None,
    scale_dtype: int = None,
    group_sizes: Optional[List[int]] = None,
    y_scale: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, Tensor? offset=None,
                                 Tensor? pertoken_scale=None, Tensor? bias=None,
                                 ScalarType? output_dtype=None) -> Tensor
    """
    import torch_npu

    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)

    if x2_dtype is not None and x2_dtype == torch_npu.int4:
        x2 = ge.Bitcast(x2, type=DataType.DT_INT4, keep_dim=True)
    
    if scale is not None and scale.dtype == DataType.DT_INT64:
        scale = ge.Bitcast(scale, type=DataType.DT_UINT64)

    out = ge.QuantBatchMatmulV3(x1,
                                x2,
                                scale,
                                offset=offset,
                                bias=bias,
                                pertoken_scale=pertoken_scale,
                                dtype=dtype,
                                transpose_x1=False,
                                transpose_x2=False)
    return out
