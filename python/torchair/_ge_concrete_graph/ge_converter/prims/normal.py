from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.normal.default)
def conveter_prims_normal_default(
    shape: Union[List[int], Tensor],
    *,
    mean: Union[Number, Tensor],
    std: Union[Number, Tensor],
    dtype: int,
    device: Device,
    requires_grad: bool,
    meta_outputs: TensorSpec = None
):
    """NB: prims::normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.normal.default ge_converter is not implemented!")
