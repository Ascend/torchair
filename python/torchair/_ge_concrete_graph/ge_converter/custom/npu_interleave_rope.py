from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_interleave_rope.default)
def conveter_aten_interleave_rope_default(
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        meta_outputs: TensorSpec = None):
    return ge.InterleaveRope(x, cos, sin)