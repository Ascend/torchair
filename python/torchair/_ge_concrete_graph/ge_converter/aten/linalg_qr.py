from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.linalg_qr.default)
def conveter_aten_linalg_qr_default(
    A: Tensor, mode: str = "reduced", meta_outputs: List[TensorSpec] = None
):
    """NB: aten::linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)"""
    raise NotImplementedError("torch.ops.aten.linalg_qr.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.linalg_qr.out)
def conveter_aten_linalg_qr_out(
    A: Tensor,
    mode: str = "reduced",
    *,
    Q: Tensor = None,
    R: Tensor = None,
    meta_outputs: List[TensorSpec] = None
):
    """NB: aten::linalg_qr.out(Tensor A, str mode="reduced", *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)"""
    raise NotImplementedError("torch.ops.aten.linalg_qr.out ge_converter is not implemented!")
