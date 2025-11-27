__all__ = []

import functools
import sys
import torch

from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode

from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager


def _pattern_extra_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    logger.debug(
                        f"Cross-stream operation detected in pattern match for Addrmsnormcast. " 
                        f"Multiple streams found: {non_default_streams}. "
                        f"Fusion is not supported for cross-stream operations."
                    )
                    return False

    if has_default and len(non_default_streams) > 0:
        logger.debug(
            f"Cross-stream operation detected in pattern match for Addrmsnormcast. " 
            f"Multiple streams found: {non_default_streams}. "
            f"Fusion is not supported for cross-stream operations."
        )
        return False

    return True


@functools.lru_cache(None)
def _register_addrmsnormcast_pattern(pattern_pass_manager: _PatternPassManager):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormcast fusion will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip fusion.')
        return

    def search_fn(x1, x2, gamma):
        y, _, xOut = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma)
        h = y.size(-1)
        y_cast = y.view(-1, h).to(torch.float32)
        return y, xOut, y_cast
    
    def replace_fn(x1, x2, gamma):
        y_cast, y, _, xOut = torch.ops.npu.npu_add_rms_norm_cast.default(
            x1, x2, gamma
        )
        h = y.size(-1)
        y_cast1 = y_cast.view(-1, h)
        return y, xOut, y_cast1
    
    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 2), device="npu", dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, device="npu", dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
            extra_check=_pattern_extra_check
        )