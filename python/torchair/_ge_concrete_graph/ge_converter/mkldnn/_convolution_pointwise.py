from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.mkldnn._convolution_pointwise.default)
def conveter_mkldnn__convolution_pointwise_default(
    X: Tensor,
    W: Tensor,
    B: Optional[Tensor],
    padding: List[int],
    stride: List[int],
    dilation: List[int],
    groups: int,
    attr: str,
    scalars: Optional[Union[List[Number], Tensor]],
    algorithm: Optional[str],
    meta_outputs: TensorSpec = None,
):
    """NB: mkldnn::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y"""
    raise NotImplementedError("torch.ops.mkldnn._convolution_pointwise.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.mkldnn._convolution_pointwise.binary)
def conveter_mkldnn__convolution_pointwise_binary(
    X: Tensor,
    other: Tensor,
    W: Tensor,
    B: Optional[Tensor],
    padding: List[int],
    stride: List[int],
    dilation: List[int],
    groups: int,
    binary_attr: str,
    alpha: Optional[Union[Number, Tensor]],
    unary_attr: Optional[str],
    unary_scalars: Optional[Union[List[Number], Tensor]],
    unary_algorithm: Optional[str],
    meta_outputs: TensorSpec = None,
):
    """NB: mkldnn::_convolution_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y"""
    raise NotImplementedError("torch.ops.mkldnn._convolution_pointwise.binary ge_converter is not implemented!")
