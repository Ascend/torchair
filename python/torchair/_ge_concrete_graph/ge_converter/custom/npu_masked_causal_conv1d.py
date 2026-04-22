from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.npu_masked_causal_conv1d.default)
def conveter_npu_masked_causal_conv1d_default(
        input: Tensor,
        weight: Tensor,
        mask: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None):
    """
    NB: func: npu_masked_causal_conv1d(Tensor input, Tensor weight, *, Tensor? mask=None) -> Tensor
    """
    return ge.MaskedCausalConv1d(input, weight, mask=mask)