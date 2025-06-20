import sys
import functools
import logging

import torch
try:
    import torch_npu
except ImportError:
    pass

from torch._dynamo.utils import counters
from torch._inductor.pattern_matcher import (
    filter_nodes,
    register_replacement,
    _return_true
)

try:
    from torch._inductor.pattern_matcher import inference_graph, training_graph
except ImportError:
    from torch._inductor.pattern_matcher import fwd_only as inference_graph
    from torch._inductor.pattern_matcher import joint_fwd_bwd as training_graph
    
from torch._inductor.fx_passes.joint_graph import patterns
from torchair._utils.npu_fx_passes.joint_graph import register_joint_graph_pass
from torchair.core.utils import logger

log = logging.getLogger(__name__)
aten = torch.ops.aten


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _romu_pattern_1(q, cos, sin):
    return (q * cos) + (rotate_half(q) * sin)


def _romu_replace_1(q, cos, sin):
    counters["inductor"]["npu_rotary_mul"] += 1
    return torch_npu.npu_rotary_mul(q, cos, sin)


@register_joint_graph_pass('npu_rotary_mul')
@functools.lru_cache(None)
def _romu_init():
    if 'torch_npu' not in sys.modules:
        logger.info(f'The npu_rotary_mul fx pass will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip npu_rotary_mul fx pass.')
        return
    device = "npu"
    
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    g = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)
    g2 = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=False)

    for pattern, replacement, args, workaround, extra_check in [
        (
            _romu_pattern_1,
            _romu_replace_1,
            [g(), g2(), g2()], 
            {},
            _return_true,
        ),
    ]:
        args = [*args, *workaround.values()]
        register_replacement(
            pattern,
            replacement,
            args,
            training_graph,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )
        register_replacement(
            pattern,
            replacement,
            args,
            inference_graph,
            patterns,
            extra_check=extra_check,
            scalar_workaround=workaround,
        )