from torchair._ge_concrete_graph.ge_converter.converter_utils import *


# 当前因为x2要求为NZ的格式，目前Support还不支持该种类型。
# ND转NZ：可通过`torch_npu.npu_format_cast(x2.contiguous(), 29)`将ND格式的x2转换为NZ格式
# @declare_supported([
#     Support(I8(8, 2048, 1024), I8_NZ(8, 1024, 7168), x1_scale=F32(8, 2048), x2_scale=BF16(7168))
# ])
@register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul_reduce_sum.default)
def conveter_npu_npu_quant_matmul_reduce_sum(
    x1: Tensor,
    x2: Tensor,
    *,
    x1_scale: Optional[Tensor] = None,
    x2_scale: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_quant_matmul_reduce_sum(
        Tensor x1, Tensor x2, *, Tensor? x1_scale=None, Tensor? x2_scale=None) -> Tensor
    """
    dims = [0]
    return ge.QuantMatmulReduceSum(
        x1,
        x2,
        dims,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        dtype=DataType.DT_BF16)
