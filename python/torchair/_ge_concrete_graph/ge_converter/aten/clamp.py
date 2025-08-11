from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(10,), None, 2),
        Support(F32(10,), 1),
        Support(F32(10,), 1, 3)
    ]
)
@register_fx_node_ge_converter(torch.ops.aten.clamp.default)
def conveter_aten_clamp_default(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor"""
    return clamp(self, max, min, meta_outputs)


def clamp(self, max_value, min_value, meta_outputs):
    if min_value is None and max_value is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    if min_value is None:
        min_value = normalize_min_value(self.dtype)
    if max_value is None:
        max_value = normalize_max_value(self.dtype)
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    min_value = dtype_promote(min_value, target_dtype=meta_outputs.dtype)
    max_value = dtype_promote(max_value, target_dtype=meta_outputs.dtype)
    return ge.ClipByValueV2(self, min_value, max_value)


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor)
def conveter_aten_clamp_Tensor(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor"""
    return clamp(self, max, min, meta_outputs)


@register_fx_node_ge_converter(torch.ops.aten.clamp.out)
def conveter_aten_clamp_out(
    self: Tensor,
    min: Optional[Union[Number, Tensor]] = None,
    max: Optional[Union[Number, Tensor]] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.clamp.out ge_converter is redundant before pytorch 2.1.0!")


@register_fx_node_ge_converter(torch.ops.aten.clamp.Tensor_out)
def conveter_aten_clamp_Tensor_out(
    self: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten.clamp.Tensor_out ge_converter is redundant before pytorch 2.1.0!")
