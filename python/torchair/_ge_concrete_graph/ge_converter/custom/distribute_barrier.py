from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.npu._npu_distribute_barrier.default)
def convert_npu_distribute_barrier(
    x_ref: Tensor,
    group: str,
    world_size: int,
    *,
    time_out: Optional[Tensor] = None,
    elastic_info: Optional[Tensor] = None,
    meta_outputs: TensorSpec = None
):

    return ge.DistributeBarrier(x_ref=x_ref,
                                time_out=time_out,
                                elastic_info=elastic_info,
                                group=group,
                                world_size=world_size)