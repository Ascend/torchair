from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2), [4]),
        Support(F16(16), [2, 8]),
        Support(U8(16), [2, 8])
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._unsafe_view.default)
def conveter_aten__unsafe_view_default(
    self: Tensor, size: Union[List[int], Tensor], meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_view(Tensor self, SymInt[] size) -> Tensor"""
    size = dtype_promote(size, target_dtype=DataType.DT_INT64)
    return ge.Reshape(self, size)


@register_fx_node_ge_converter(torch.ops.aten._unsafe_view.out)
def conveter_aten__unsafe_view_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_unsafe_view.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._unsafe_view.out ge_converter is not implemented!")
