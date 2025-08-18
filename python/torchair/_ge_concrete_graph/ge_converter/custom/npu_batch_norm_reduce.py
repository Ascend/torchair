from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu.batch_norm_reduce.default)
def conveter_npu_batch_norm_reduce_default(
    self: Tensor,
    eps: float,
    meta_outputs: TensorSpec = None
):  
    raise NotImplementedError("torch.ops.npu.batch_norm_reduce.default ge_converter is not implemented!")
