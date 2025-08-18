from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_batch_gather_matmul.default)
def conveter_npu_batch_gather_matmul_default(
    y: Tensor,
    x: Tensor,
    weight_b: Tensor,
    indices: Tensor,
    weight_a: Optional[Tensor] = None,
    layer_idx: int = 0,
    scale: float = 1e-3,
    y_offset: int = 0,
    y_slice_size: int = -1,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_batch_gather_matmul(Tensor y, Tensor x, Tensor weight_b, Tensor indices, Tensor? weight_a=None,
                                    int layer_idx=0, float scale=1e-3, int y_offset=0, int y_slice_size=-1) -> Tensor"""
    return ge.AddLora(y, x, weightA=weight_a, weightB=weight_b, indices=indices,
                      layer_idx=layer_idx, scale=scale, y_offset=y_offset, y_slice_size=y_slice_size)
