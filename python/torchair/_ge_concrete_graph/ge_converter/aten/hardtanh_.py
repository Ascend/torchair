from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.aten.hardtanh_.default)
def conveter_aten_hardtanh__default(
    self: Tensor,
    min_val: Union[Number, Tensor] = -1,
    max_val: Union[Number, Tensor] = 1,
    meta_outputs: TensorSpec = None,
):
    """NB: aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)"""
    raise NotImplementedError("torch.ops.aten.hardtanh_.default ge_converter is not implemented!")
