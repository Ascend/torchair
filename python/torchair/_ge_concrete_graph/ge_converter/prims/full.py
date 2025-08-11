from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.full.default)
def conveter_prims_full_default(
    shape: Union[List[int], Tensor],
    fill_value: Union[Number, Tensor],
    *,
    dtype: int,
    device: Device,
    requires_grad: bool,
    meta_outputs: TensorSpec = None
):
    """NB: prims::full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.full.default ge_converter is not implemented!")
