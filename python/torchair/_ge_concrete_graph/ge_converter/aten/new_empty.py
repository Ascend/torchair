from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.new_empty.default)
def conveter_aten_new_empty_default(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        dtype = self.dtype
    else:
        dtype = torch_type_to_ge_type(dtype)
    size = dtype_promote(size, target_dtype=DataType.DT_INT32)
    if layout is not None and layout != torch.strided:
        raise RuntimeError(f"torch.ops.aten.new_empty.default layout only support torch.strided, but now is {layout}!")
    return ge.Empty(size, dtype=dtype)


@register_fx_node_ge_converter(torch.ops.aten.new_empty.out)
def conveter_aten_new_empty_out(
    self: Tensor,
    size: Union[List[int], Tensor],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::new_empty.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.new_empty.out ge_converter is not supported!")
