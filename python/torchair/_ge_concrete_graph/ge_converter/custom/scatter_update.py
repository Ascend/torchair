from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.scatter_update.default)
def conveter_npu_scatter_update_default(
    data: Tensor,
    indices: Tensor,
    updates: Tensor = 0,
    axis: int = 0,
    meta_outputs: TensorSpec = None,
):
    """NB: func: scatter_update(Tensor data, Tensor indices, Tensor updates, int axis) -> Tensor"""
    """
    Warning: kernel [scatter_update] is a out-of-place op, but it is supported by another in-place op cann.Scatter.
    This current usage may cause the input to be changed unexpectedly, 
    and the caller needs to pay attention to this feature.
    """

    copy = ge.TensorMove(data)
    return ge.Scatter(copy, indices, updates, reduce="update", axis=axis)
