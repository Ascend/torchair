from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.scalar_tensor.default)
def conveter_prims_scalar_tensor_default(
    s: Union[Number, Tensor],
    *,
    dtype: Optional[int] = None,
    device: Optional[Device] = None,
    meta_outputs: TensorSpec = None
):
    """NB: prims::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.scalar_tensor.default ge_converter is not implemented!")
