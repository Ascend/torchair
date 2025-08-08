from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.higher_order.run_with_rng_state)
def converter_run_with_rng_state_default(
    rng_state: Tensor,
    op: OpOverload,
    *args,
    meta_outputs: TensorSpec = None,
    **kwargs
):
    fn = get_checkpoint_func(op)
    _, result = fn(*args, meta_outputs=meta_outputs, **kwargs, rng_state=rng_state)
    return result