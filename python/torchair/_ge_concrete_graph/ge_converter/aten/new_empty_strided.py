from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F32(96, 512, 1, 1), [0, 2], [1, 1])
])
@register_fx_node_ge_converter(torch.ops.aten.new_empty_strided.default)
def conveter_aten_new_empty_strided_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = self.dtype
    else:
        dtype = torch_type_to_ge_type(dtype)
    size = dtype_promote(size, target_dtype=DataType.DT_INT32)
    if layout is not None and layout != torch.strided:
        raise AssertionError(
            "torch.ops.aten.new_empty_strided.default ge_converter is only supported on dense tensor now!")
    result = ge.Empty(size, dtype=dtype)
    result = ge.AsStrided(result, size, stride, 0)
    return result


@register_fx_node_ge_converter(torch.ops.aten.new_empty_strided.out)
def conveter_aten_new_empty_strided_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    stride: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_empty_strided.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)"""
    raise AssertionError(
        "torch.ops.aten.new_empty_strided.out is redundant before pytorch 2.1.0, "
        "might be supported in furture version.")
