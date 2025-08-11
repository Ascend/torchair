from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(6, 4), [6, 4], 30, dtype=torch.float16),
        Support(F32(6, 4), [6, 4], 30),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.new_full.default)
def conveter_aten_new_full_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is not None:
        fill_value = dtype_promote(fill_value, target_dtype=dtype)
    else:
        fill_value = dtype_promote(fill_value, target_dtype=self.dtype)
    if layout is not None and layout != torch.strided:
        raise RuntimeError("torch.ops.aten.new_full.default ge_converter is only supported on dense tensor now!")
    return ge.Fill(size, fill_value)


@register_fx_node_ge_converter(torch.ops.aten.new_full.out)
def conveter_aten_new_full_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_full.out(Tensor self, SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.new_full.out ge_converter is not supported!")
