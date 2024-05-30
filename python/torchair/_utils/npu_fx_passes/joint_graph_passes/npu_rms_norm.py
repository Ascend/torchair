import sys
import functools
import logging

import torch
import torch.nn as nn

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


def _npu_rms_norm_pattern_1(hidden_states, weight, epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states.to(input_dtype)


def _npu_rms_norm_replace_1(hidden_states, weight, epsilon):
    counters["inductor"]["npu_rms_norm"] += 1
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


@register_joint_graph_pass('npu_rms_norm')
@functools.lru_cache(None)
def _npu_rms_norm_init():
    if 'torch_npu' not in sys.modules:
        logger.info(f'The npu_rms_norm fx pass will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip npu_rms_norm fx pass.')
        return
    device = "npu"
    
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    h1 = functools.partial(torch.empty, (4, 2048, 4096), device=device, requires_grad=True)
    w1 = functools.partial(torch.empty, (4096,), device=device, requires_grad=True)
    d1 = {"epsilon": 1e-5}

    for pattern, replacement, args, workaround, extra_check in [
        (
            _npu_rms_norm_pattern_1,
            _npu_rms_norm_replace_1,
            [h1(), w1()], 
            d1,
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