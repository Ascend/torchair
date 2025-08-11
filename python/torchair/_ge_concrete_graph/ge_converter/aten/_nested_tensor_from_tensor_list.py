from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten._nested_tensor_from_tensor_list.default)
def conveter_aten__nested_tensor_from_tensor_list_default(
    list: List[Tensor],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::_nested_tensor_from_tensor_list(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten._nested_tensor_from_tensor_list.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten._nested_tensor_from_tensor_list.out)
def conveter_aten__nested_tensor_from_tensor_list_out(
    list: List[Tensor],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_nested_tensor_from_tensor_list.out(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten._nested_tensor_from_tensor_list.out ge_converter is not implemented!")
