from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.minimum_value.default)
def conveter_prims_minimum_value_default(dtype: int, meta_outputs: TensorSpec = None):
    """NB: prims::minimum_value(ScalarType dtype) -> Scalar"""
    raise NotImplementedError("torch.ops.prims.minimum_value.default ge_converter is not implemented!")
