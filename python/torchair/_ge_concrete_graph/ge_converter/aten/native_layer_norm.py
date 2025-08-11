from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def _fill_if_none(
    tensor: Optional[Tensor],
    normalized_shape: Union[List[int], Tensor],
    value: float,
    dtype: DataType
) -> Tensor:
    if tensor is None:
        if isinstance(normalized_shape, Tensor):
            dtype_promote(normalized_shape, target_dtype=DataType.DT_INT32)
            return ge.Fill(normalized_shape, ge.Const(value, dtype))
        else:
            return ge.Fill(ge.Const(normalized_shape, dtype=DataType.DT_INT32), ge.Const(value, dtype))
    return tensor


@declare_supported(
    [
        Support(F16(2, 1024), [1024], F16(1024), F16(1024), 1e-5),
        Support(F16(2, 1024), [1024], None, None, 1e-5),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm.default)
def conveter_aten_native_layer_norm_default(
    input: Tensor,
    normalized_shape: Union[List[int], Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)"""
    weight = _fill_if_none(weight, normalized_shape, 1., input.dtype)
    bias = _fill_if_none(bias, normalized_shape, 0., input.dtype)

    return ge.LayerNormV4(input, normalized_shape, gamma=weight, beta=bias, epsilon=eps)


@register_fx_node_ge_converter(torch.ops.aten.native_layer_norm.out)
def conveter_aten_native_layer_norm_out(
    input: Tensor,
    normalized_shape: Union[List[int], Tensor],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    *,
    out0: Tensor = None,
    out1: Tensor = None,
    out2: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::native_layer_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))"""
    raise RuntimeError("torch.ops.aten.native_layer_norm.out ge_converter is "
                       "redundant before pytorch 2.1.0, might be supported in future version.")
