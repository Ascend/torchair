from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_geglu.default)
def conveter_npu_geglu_default(
        self: Tensor,
        dim: int = -1,
        approximate: int = 1,
        activate_left: bool = False,
        meta_outputs: TensorSpec = None):
    """ NB: npu::npu_geglu(Tensor self, int dim=-1, int approximate=1, bool activate_left=False) -> (Tensor, Tensor) """
    return ge.GeGluV2(self, dim=dim, approximate=approximate, activate_left=activate_left)