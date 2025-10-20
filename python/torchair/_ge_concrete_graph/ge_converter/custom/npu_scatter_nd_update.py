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


@register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update_.default)
def conveter_npu_scatter_nd_update__default(
        self: Tensor,
        indices: Tensor,
        updates: Tensor,
        meta_outputs: TensorSpec = None,
):
    """
    NB: func: npu_scatter_nd_update_(Tensor(a!) input, Tensor indices, Tensor updates) -> Tensor(a!)
    """

    """
    The converter for inplace operators is generally not necessary, 
    because all inplace operators become non_inplace operators after functionalization.
    Adding converters to those inplace operators is due to the implementation of some re-inplace pass, 
    which pass can transfer some non_inplace operators to the original inplace operators.
    """

    return ge.ScatterNdUpdate(self, indices, updates)
