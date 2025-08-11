from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.uniform.default)
def conveter_prims_uniform_default(
    shape: Union[List[int], Tensor],
    *,
    low: Union[Number, Tensor],
    high: Union[Number, Tensor],
    dtype: int,
    device: Device,
    meta_outputs: TensorSpec = None
):
    """NB: prims::uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.uniform.default ge_converter is not implemented!")
