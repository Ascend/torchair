from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.empty_strided.default)
def conveter_prims_empty_strided_default(
    shape: Union[List[int], Tensor],
    strides: Union[List[int], Tensor],
    *,
    dtype: int,
    device: Device,
    requires_grad: bool,
    meta_outputs: TensorSpec = None
):
    """NB: prims::empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.empty_strided.default ge_converter is not implemented!")
