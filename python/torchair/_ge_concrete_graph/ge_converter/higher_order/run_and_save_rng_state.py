from torchair._ge_concrete_graph.ge_converter.converter_utils import *


@register_fx_node_ge_converter(torch.ops.higher_order.run_and_save_rng_state)
def converter_run_and_save_rng_state_default(
    op: OpOverload,
    *args,
    meta_outputs: TensorSpec = None,
    **kwargs
):
    fn = get_checkpoint_func(op)
    seed_and_offset, result = fn(*args, meta_outputs=meta_outputs, **kwargs)
    seed, offset = seed_and_offset
    rng_state = ge.ConcatV2([seed, offset], 0, N=2)
    return rng_state, result
