from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(4, 2, 128, 64), F32(4, 2, 128, 128), F32(4, 2, 128, 64), [True, True]),
        Support(F32(4, 2, 128, 128), F32(4, 2, 128, 64), F32(4, 2, 64, 128), [True, True]),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.matmul_backward.default)
def conveter_aten_matmul_backward_default(
    grad: Tensor, 
    self: Tensor, 
    other: Tensor, 
    mask: List[bool], 
    meta_outputs: TensorSpec = None
):
    """NB: aten::matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)"""
    if len(mask) != 2 or not mask[0] or not mask[1]:
        raise NotImplementedError("matmul_backward currently only support mask == [True, True]")
    self_perm_list = [i for i in range(self.rank)]
    self_perm_list[-1], self_perm_list[-2] = self_perm_list[-2], self_perm_list[-1]

    other_perm_list = [i for i in range(other.rank)]
    other_perm_list[-1], other_perm_list[-2] = other_perm_list[-2], other_perm_list[-1]
    
    self_perm_list, other_perm_list = dtype_promote(self_perm_list, other_perm_list, target_dtype=DataType.DT_INT64)
    grad_self = ge.BatchMatMul(grad, ge.Transpose(other, other_perm_list))
    grad_other = ge.BatchMatMul(ge.Transpose(self, self_perm_list), grad)
    return grad_self, grad_other

