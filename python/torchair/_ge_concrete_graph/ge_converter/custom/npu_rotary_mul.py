from torchair._ge_concrete_graph.ge_converter.converter_utils import *
from torchair._utils.check_platform import is_not_support


@declare_supported([
    Support(F32(2, 8192, 5, 128), F32(1, 8192, 1, 128), F32(1, 8192, 1, 128)),
    Support(F16(2, 8192, 5, 128), F16(1, 8192, 1, 128), F16(1, 8192, 1, 128)),
])
@register_fx_node_ge_converter(torch.ops.npu.npu_rotary_mul.default)
def conveter_npu_rotary_mul_default(
    self: Tensor,
    r1: Tensor,
    r2: Tensor,
    rotary_mode: str = 'half',
    meta_outputs: TensorSpec = None
):
    """NB: npu::npu_rotary_mul(Tensor self, Tensor r1, Tensor r2) -> Tensor"""
    if is_not_support():
        modes = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
        if rotary_mode not in modes:
            raise NotImplementedError("rotary_mode only support half/interleave/quarter/interleave-half now!")
        mode = modes[rotary_mode]
        return ge.RotaryPositionEmbedding(self, r1, r2, mode=mode)
    else:
        return ge.RotaryMul(self, r1, r2)
