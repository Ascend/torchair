from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.addbmm_.default)
def conveter_aten_addbmm__default(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: Union[Number, Tensor] = 1,
    alpha: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None
):
    """NB: aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.addbmm_.default ge_converter is not implemented!")
