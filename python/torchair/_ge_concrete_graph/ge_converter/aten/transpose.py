from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 3), 0, 1),
        Support(I8(2, 3), 0, 1),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.transpose.int)
def conveter_aten_transpose_int(
    self: Tensor, dim0: int, dim1: int, meta_outputs: TensorSpec = None
):
    """NB: aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)"""
    perm_list = [i for i in range(self.rank)]
    perm_list[dim0], perm_list[dim1] = perm_list[dim1], perm_list[dim0]
    perm_list = dtype_promote(perm_list, target_dtype=DataType.DT_INT64)
    return ge.Transpose(self, perm_list)


@register_fx_node_ge_converter(torch.ops.aten.transpose.Dimname)
def conveter_aten_transpose_Dimname(
    self: Tensor, dim0: str, dim1: str, meta_outputs: TensorSpec = None
):
    """NB: aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> Tensor(a)"""
    raise NotImplementedError("torch.ops.aten.transpose.Dimname ge_converter is not implemented!")
