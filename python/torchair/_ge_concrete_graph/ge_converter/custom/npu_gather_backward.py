from typing import (
    List,
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType


@register_fx_node_ge_converter(torch.ops.npu.npu_gather_backward.default)
def conveter_npu_gather_backward_default(
        grad: Tensor,
        self_size: List[int],
        dim: int,
        index: Tensor,
        sparse_grad: bool,
        meta_outputs: TensorSpec = None):
    """
    NB: npu::npu_gather_backward(Tensor grad, SymInt[] self_size, int dim, Tensor index, bool sparse_grad) -> Tensor
    """
    zero_out = ge.Fill(self_size, ge.Cast(0, dst_type=grad.dtype))
    return ge.ScatterElements(zero_out, index, grad, axis=dim, reduction='add')
