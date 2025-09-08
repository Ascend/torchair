import torch
from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._ge_concrete_graph import ge_apis as ge
from torchair.ge._ge_graph import Tensor, TensorSpec, torch_type_to_ge_type


@register_fx_node_ge_converter(torch.ops.prims.convert_element_type.default)
def conveter_prims_convert_element_type_default(
    a: Tensor, dtype: int, meta_outputs: TensorSpec = None
):
    """NB: prims::convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""
    return ge.Cast(a, dst_type=torch_type_to_ge_type(dtype))
