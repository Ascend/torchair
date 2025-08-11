from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(0.125, 0.875, 7, dtype=torch.float32),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.linspace.default)
def conveter_aten_linspace_default(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    *,
    dtype: Optional[int] = None,
    layout: Optional[int] = None,
    device: Optional[Device] = None,
    pin_memory: Optional[bool] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"""
    if steps < 0:
        raise AssertionError("number of steps must be non-negative")
    if layout is not None and layout != torch.strided:
        raise RuntimeError("torch.ops.aten.linspace.default ge_converter is only supported on dense tensor now!")
    result = ge.LinSpace(start, end, steps)
    if dtype is not None:
        result = dtype_promote(result, target_dtype=dtype)
    return result


@register_fx_node_ge_converter(torch.ops.aten.linspace.out)
def conveter_aten_linspace_out(
    start: Union[Number, Tensor],
    end: Union[Number, Tensor],
    steps: int,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.linspace.out ge_converter is not supported!")
