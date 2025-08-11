from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.zeros.names)
def conveter_aten_zeros_names(
    size: List[int],
    *,
    names: Optional[List[str]],
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    raise RuntimeError("torch.ops.aten.zeros.names ge_converter is not supported!")


@declare_supported(
    [
        Support((2, 3)),
        Support((2, 3), dtype=torch.int),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.zeros.default)
def conveter_aten_zeros_default(
    size: Union[List[int], Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is None:
        return ge.Fill(size, 0.)
    else:
        return ge.Fill(size, ge.Cast(0., dst_type=torch_type_to_ge_type(dtype)))


@register_fx_node_ge_converter(torch.ops.aten.zeros.names_out)
def conveter_aten_zeros_names_out(
    size: List[int],
    *,
    names: Optional[List[str]],
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.zeros.names_out ge_converter is not supported!")


@register_fx_node_ge_converter(torch.ops.aten.zeros.out)
def conveter_aten_zeros_out(
    size: Union[List[int], Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.zeros.out ge_converter is not supported!")
