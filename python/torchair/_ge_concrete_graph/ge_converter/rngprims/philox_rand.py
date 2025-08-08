from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.rngprims.philox_rand.default)
def conveter_rngprims_philox_rand_default(
    size: Union[List[int], Tensor],
    seed: Tensor,
    offset: Tensor,
    stride: Optional[List[int]],
    device: Optional[Device] = None,
    dtype: Optional[int] = None,
    meta_outputs: List[TensorSpec] = None,
):
    """NB: rngprims::philox_rand(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"""
    raise NotImplementedError("torch.ops.rngprims.philox_rand.default ge_converter is not implemented!")
