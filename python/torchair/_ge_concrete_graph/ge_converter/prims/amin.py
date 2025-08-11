from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.amin.default)
def conveter_prims_amin_default(
    inp: Tensor,
    dims: Optional[List[int]],
    *,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: prims::amin(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.amin.default ge_converter is not implemented!")
