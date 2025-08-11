from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addcmul_.default)
def conveter_aten_addcmul__default(
    self: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addcmul_.default ge_converter is not implemented!")
