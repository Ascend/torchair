from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@declare_supported([
    Support(F16(64, 16384, 1280), F16(64, 16384, 1280), I64(1, 16384), I64(1, 16384)),
])
@register_fx_node_ge_converter(torch.ops.npu_inference.npu_tome_merge.default)
def conveter_npu_tome_merge_default(
    token_a: Tensor,
    token_b: Tensor,
    token_indice: Tensor,
    arg_max: Tensor,
    top_rate: float = 0.5,
    meta_outputs: List[TensorSpec] = None
):
    return ge.TomeMerge(token_a, token_b, token_indice, arg_max, top_rate)