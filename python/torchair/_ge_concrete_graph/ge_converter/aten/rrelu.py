from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.rrelu.default)
def conveter_aten_rrelu_default(
    self: Tensor,
    lower: Union[Number, Tensor] = 0.125,
    upper: Union[Number, Tensor] = 0.3333333333333333,
    training: bool = False,
    generator: Optional[Generator] = None,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor"""
    raise NotImplementedError("torch.ops.aten.rrelu.default ge_converter is not implemented!")
