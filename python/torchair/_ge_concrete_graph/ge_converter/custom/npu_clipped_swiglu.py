from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_clipped_swiglu.default)
def conveter_aten_clipped_swiglu_default(
        x: Tensor,
        group_index: Tensor = None,
        dim: int = -1,
        alpha: float = 1.702,
        limit: float = 7.0,
        bias: float = 1.0, 
        interleaved: bool = True,
        meta_outputs: TensorSpec = None):
    """
    NB: func: npu_clipped_swiglu(Tensor self, *, Tensor? group_index=None, int dim=-1,
    float alpha=1.702, float limit=7.0, float bias=1.0, bool interleaved=True) -> Tensor
    """
    return ge.ClippedSwiglu(x, group_index=group_index, dim=dim, alpha=alpha, limit=limit, bias=bias, interleaved=interleaved)