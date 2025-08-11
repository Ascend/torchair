from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor)
def conveter_aten_repeat_interleave_Tensor(
    repeats: Tensor, *, output_size: Optional[int] = None, meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_Tensor)
def conveter_aten_repeat_interleave_self_Tensor(
    self: Tensor,
    repeats: Tensor,
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_Tensor ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.self_int)
def conveter_aten_repeat_interleave_self_int(
    self: Tensor,
    repeats: Union[int, Tensor],
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, int? output_size=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.self_int ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.repeat_interleave.Tensor_out)
def conveter_aten_repeat_interleave_Tensor_out(
    repeats: Tensor,
    *,
    output_size: Optional[int] = None,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::repeat_interleave.Tensor_out(Tensor repeats, *, int? output_size=None, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.repeat_interleave.Tensor_out ge_converter is not implemented!")
