from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.scalar_tensor.default)
def conveter_aten_scalar_tensor_default(
    s: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if dtype is not None:
        # When a dtype is specified, ensure that the input scalar can be converted to the required dtype by inserting
        # a ge.Cast or ge.Const node if necessary.
        return dtype_promote(s, target_dtype=dtype)
    if isinstance(s, Tensor):
        return s
    return ge.Const(s)


@register_fx_node_ge_converter(torch.ops.aten.scalar_tensor.out)
def conveter_aten_scalar_tensor_out(
    s: Union[Number, Tensor], *, out: Tensor = None, meta_outputs: TensorSpec = None
):
    """NB: aten::scalar_tensor.out(Scalar s, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.scalar_tensor.out ge_converter is not implemented!")
