from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.im2col.default)
def conveter_aten_im2col_default(
    self: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.im2col.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.im2col.out)
def conveter_aten_im2col_out(
    self: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.im2col.out ge_converter is not implemented!")
