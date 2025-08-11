from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.col2im.default)
def conveter_aten_col2im_default(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    meta_outputs: TensorSpec = None,
):
    """NB: aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.col2im.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.aten.col2im.out)
def conveter_aten_col2im_out(
    self: Tensor,
    output_size: Union[List[int], Tensor],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
    *,
    out: Tensor = None,
    meta_outputs: TensorSpec = None
):
    """NB: aten::col2im.out(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.col2im.out ge_converter is not implemented!")
