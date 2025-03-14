from typing import (
    Any,
    Optional
)

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter, \
                                                       torch_type_to_ge_type
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import F32, I8, I32, Support


@declare_supported([
    Support(I32(24, 64), F32(32), None, None),
    Support(I32(24, 64), F32(32), None, None, 1),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dequant_bias.default)
def convert_npu_dequant_bias_default(
    x: Tensor,
    weight_scale: Tensor,
    activate_scale: Optional[Tensor],
    bias: Optional[Tensor],
    *,
    output_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_dequant_bias(Tensor x, Tensor weight_scale, Tensor? activation_scale, Tensor? bias, *,
                                 ScalarType? output_dtype=None) -> Tensor"""

    attr_output_type = DataType.DT_FLOAT16
    if output_dtype is not None:
        attr_output_type = torch_type_to_ge_type(output_dtype)
    
    return ge.dequant_bias(x, weight_scale, activate_scale, bias, output_dtype=attr_output_type)