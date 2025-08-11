from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.iota.default)
def conveter_prims_iota_default(
    length: Union[int, Tensor],
    *,
    start: Union[int, Tensor],
    step: Union[int, Tensor],
    dtype: int,
    device: Device,
    requires_grad: bool,
    meta_outputs: TensorSpec = None
):
    """NB: prims::iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.iota.default ge_converter is not implemented!")
