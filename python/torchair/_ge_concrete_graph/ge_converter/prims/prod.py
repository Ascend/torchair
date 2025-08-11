from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.prod.default)
def conveter_prims_prod_default(
    inp: Tensor,
    dims: Optional[List[int]],
    *,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: prims::prod(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.prod.default ge_converter is not implemented!")
