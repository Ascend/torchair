from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)
def conveter_npu_scatter_nd_update_default(
    self: Tensor,
    indices: Tensor,
    updates: Tensor,
    meta_outputs: TensorSpec = None,
):
    """NB: func: scatter_nd_update(Tensor self, Tensor indices, Tensor updates) -> Tensor"""

    copy = ge.TensorMove(self)
    return ge.ScatterNdUpdate(copy, indices, updates)
