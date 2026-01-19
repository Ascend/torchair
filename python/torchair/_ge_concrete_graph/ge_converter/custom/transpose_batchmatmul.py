from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_transpose_batchmatmul.default)
def conveter_npu_npu_transpose_batchmatmul(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    scale: Optional[Tensor] = None,
    perm_x1: Optional[List[int]] = [0, 1, 2],
    perm_x2: Optional[List[int]] = [0, 1, 2],
    perm_y: Optional[List[int]] = [1, 0, 2],
    batch_split_factor: Optional[int] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: npu::npu_transpose_batchmatmul(Tensor input, Tensor weight, *, Tensor? bias=None,
    Tensor? scale=None, int[]? perm_x1=None, int[]? perm_x2=None, int[]? perm_y=None,
    int? batch_split_factor=1) -> Tensor
    """

    return ge.TransposeBatchMatMul(input, weight, bias=bias, scale=scale,
                                   perm_x1=perm_x1, perm_x2=perm_x2, perm_y=perm_y,
                                   enable_hf32=torch.npu.matmul.allow_hf32, batch_split_factor=batch_split_factor)
