from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.prims.convert_element_type.default)
def conveter_prims_convert_element_type_default(
    a: Tensor, dtype: int, meta_outputs: TensorSpec = None
):
    """NB: prims::convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""
    raise NotImplementedError("torch.ops.prims.convert_element_type.default ge_converter is not implemented!")
