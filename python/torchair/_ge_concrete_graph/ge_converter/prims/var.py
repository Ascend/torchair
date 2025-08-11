from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.var.default)
def conveter_prims_var_default(
    inp: Tensor,
    dims: Optional[List[int]],
    *,
    correction: float,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: prims::var(Tensor inp, int[]? dims, *, float correction, ScalarType? output_dtype=None) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.var.default ge_converter is not implemented!")
