from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported(
    [
        Support(F32(2, 2), dim=0, half_to_float=False),
        Support(F32(2, 2), dim=1, half_to_float=True),
    ]
)
@register_fx_node_ge_converter(torch.ops.aten._log_softmax.default)
def conveter_aten__log_softmax_default(
    self: Tensor, dim: int, half_to_float: bool, meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    self = dtype_promote(self, target_dtype=meta_outputs.dtype)
    output = ge.LogSoftmaxV2(self, axes=[dim])
    return output


@register_fx_node_ge_converter(torch.ops.aten._log_softmax.out)
def conveter_aten__log_softmax_out(
    self: Tensor,
    dim: int,
    half_to_float: bool,
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)"""
    raise RuntimeError("torch.ops.aten._log_softmax.out ge_converter is not supported!")
