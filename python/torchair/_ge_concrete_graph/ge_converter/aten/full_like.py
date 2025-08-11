from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(2, 1), 1),
])
@register_fx_node_ge_converter(torch.ops.aten.full_like.default)
def conveter_aten_full_like_default(
    self: Tensor,
    fill_value: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    memory_format: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor"""
    dims = ge.Shape(self)
    self = dtype_promote(self, target_dtype=dtype) if (dtype and torch_type_to_ge_type(dtype) != self.dtype) else self
    dst_dtype = torch_type_to_ge_type(dtype) if dtype else self.dtype

    # layout, device, pin_memory, memory_format do not affect graph
    return ge.Fill(dims, ge.Cast(fill_value, dst_type=dst_dtype))


@register_fx_node_ge_converter(torch.ops.aten.full_like.out)
def conveter_aten_full_like_out(
    self: Tensor,
    fill_value: Union[Number, Tensor],
    *,
    memory_format: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::full_like.out(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.full_like.out ge_converter is not supported!")
