from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 2), [I64(1, 2)]),
    Support(F32(2, 2), [I64(1, 2), I64(1, 2)]),
])
@register_fx_node_ge_converter(torch.ops.aten._unsafe_index.Tensor)
def conveter_aten__unsafe_index_Tensor(
    self: Tensor, indices: List[Optional[Tensor]], meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"""
    if self.dtype in [DataType.DT_BOOL, DataType.DT_UINT8]:
        raise NotImplementedError("_unsafe_index.Tensor currently not support dtype Bool or Uint8.")
    mask = [1 if indice else 0 for indice in indices]
    indices = [i for i in indices if i]
    return ge.IndexByTensor(self, indices, indices_mask=mask)
