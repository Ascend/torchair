from typing import List, Optional
import torch
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.supported_declaration import F16, BF16, Support


@declare_supported([
    Support(F16(64, 16384, 16384)),
    Support(BF16(64, 16384, 16384)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_dynamic_quant.default)
def conveter_npu_dynamic_quant_default(
    input_data: Tensor,
    smooth_scales: Optional[Tensor] = None,
    meta_outputs: List[TensorSpec] = None
):
    return ge.DynamicQuant(input_data, smooth_scales)