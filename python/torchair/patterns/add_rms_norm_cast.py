__all__ = []

import functools
import operator
import sys
import torch

from torch._inductor.pattern_matcher import Match, MultiOutputPattern, CallFunction, KeywordArg, Ignored
from torch._subclasses.fake_tensor import FakeTensorMode

from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager


def _pattern_extra_check(match: Match) -> bool:
    """
    Check if the dimension argument in y.size(-1) is exactly -1 (last dimension of tensor y).
    """
    view_node = None
    x1_node = None

    for node in match.nodes:
        if node.target == torch.ops.aten.view.default:
            view_node = node
        elif node.target == torch.ops.npu.npu_add_rms_norm.default and len(node.args) >= 1:
            x1_node = node.args[0]

    if not view_node or not x1_node:
        logger.debug("Extra check failed for addrmsnormcast, view_node or x1_node is none.")
        return False

    view_args = view_node.args[1]
    if not isinstance(view_args, (list, tuple)):
        logger.debug("Extra check failed for addrmsnormcast, view_node args is not list or tuple.")
        return False

    x1_last_dim = None
    if hasattr(x1_node, 'meta') and 'tensor_meta' in x1_node.meta:
        x1_shape = x1_node.meta['tensor_meta'].shape
        if len(x1_shape) >= 1:
            x1_last_dim = x1_shape[-1]

    view_arg_val = None
    view_arg_first = view_args[1]
    if hasattr(view_arg_first, 'meta') and 'val' in view_arg_first.meta:
        view_arg_val = view_arg_first.meta['val']
    else:
        view_arg_val = view_arg_first

    return view_arg_val and str(view_arg_val) == str(x1_last_dim)


def _build_search_pattern(is_use_aten_copy: bool) -> MultiOutputPattern:
    """
    Multi-output matching pattern equivalent to the operator combination in search_fn:
    
    def search_fn(x1, x2, gamma):
        y, _, xOut = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma)
        h = y.size(-1)
        y_cast = y.view(-1, h).to(torch.float32)
        return y, xOut, y_cast
    
    - add_rms_norm_output0: 1st output of npu_add_rms_norm.default (corresponds to y in search_fn)
    - add_rms_norm_output2: 3rd output of npu_add_rms_norm.default (corresponds to xOut in search_fn)
    - cast_output: Output of y.view(-1, h).to(torch.float32) (h = y.size(-1), corresponds to y_cast in search_fn)
    """
    npu_add_rms_norm_func = CallFunction(
        torch.ops.npu.npu_add_rms_norm.default, 
        KeywordArg('x1'), 
        KeywordArg('x2'), 
        KeywordArg('gamma'), 
        _users=2
    )

    add_rms_norm_output0 = CallFunction(
        operator.getitem,
        npu_add_rms_norm_func,
        0, 
        _users=2
    )

    add_rms_norm_output2 = CallFunction(
        operator.getitem,
        npu_add_rms_norm_func,
        2
    )

    cast_output = CallFunction(
        torch.ops.npu._npu_dtype_cast.default,
        CallFunction(
            torch.ops.aten.view.default,
            add_rms_norm_output0,
            [-1, Ignored()]
        ),
        torch.float32
    )

    if is_use_aten_copy:
        cast_output = CallFunction(
            torch.ops.aten._to_copy.default,
            CallFunction(
                torch.ops.aten.view.default,
                add_rms_norm_output0,
                [-1, Ignored()]
            ),
            dtype=torch.float32
        )

    return MultiOutputPattern(
        [
            add_rms_norm_output0,
            add_rms_norm_output2,
            cast_output
        ]
    )


@functools.lru_cache(None)
def _register_addrmsnormcast_pattern(pattern_pass_manager: _PatternPassManager):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormcast fusion will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip fusion.')
        return

    def search_fn(x1, x2, gamma):
        pass
    
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
        input_tensor = functools.partial(torch.empty, (1, 1, 2), dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
            extra_check=_pattern_extra_check,
            search_fn_pattern=_build_search_pattern(True)
        )
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
            extra_check=_pattern_extra_check,
            search_fn_pattern=_build_search_pattern(False)
        )