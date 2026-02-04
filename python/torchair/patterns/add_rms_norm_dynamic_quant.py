__all__ = []

import functools
import operator
import sys
import torch

from torch._inductor.pattern_matcher import Match, MultiOutputPattern, CallFunction, KeywordArg, Ignored
from torch._subclasses.fake_tensor import FakeTensorMode

from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager


def _check_view_shape(match: Match) -> bool:
    """
    Check if the aten.view.default operator applied to the target tensor is equivalent 
    to y.flatten(0, 1) (merging the first two dimensions of tensor y).
    """
    view_node = None
    x1_node = None
    y_node = None

    for node in match.nodes:
        if node.target == torch.ops.npu.npu_add_rms_norm.default and len(node.args) >= 1:
            x1_node = node.args[0]
        elif (node.target == operator.getitem and len(node.args) >= 2):
            if (node.args[0].target == torch.ops.npu.npu_add_rms_norm.default and
              node.args[1] == 0):
                y_node = node
        elif (node.target == torch.ops.aten.view.default and len(node.args) >= 2 and
              node.args[0] == y_node):
            view_node = node

    if not view_node or not x1_node or not y_node:
        logger.debug("Extra check failed for addrmsnormdynamicquant, view_node or x1_node or y_node is none.")
        return False

    x1_dim = 0
    y_dim = 0
    y_shape = None
    if hasattr(x1_node, 'meta') and 'tensor_meta' in x1_node.meta:
        x1_dim = len(x1_node.meta['tensor_meta'].shape)
    if hasattr(y_node, 'meta') and 'tensor_meta' in y_node.meta:
        y_shape = y_node.meta['tensor_meta'].shape
        y_dim = len(y_shape)

    if x1_dim != y_dim:
        logger.debug("Extra check failed for addrmsnormdynamicquant, x1_dim is not equals to y_dim.")
        return False

    view_size = view_node.args[1]
    if not isinstance(view_size, (list, tuple)):
        logger.debug("Extra check failed for addrmsnormdynamicquant, view_node args is not list or tuple.")
        return False
    
    isShapeMatch = len(view_size) == x1_dim - 1
    if hasattr(view_size[0], 'name'):
        isShapeMatch = view_size[0].name.startswith('mul') and isShapeMatch
    elif y_dim >= 2:
        isShapeMatch = (view_size[0] == y_shape[0] * y_shape[1]) and isShapeMatch

    logger.debug(
        f"End extra check for addrmsnormdynamicquant: {isShapeMatch}. "
    )
    return isShapeMatch


@functools.lru_cache(None)
def _register_addrmsnormdynamicquant_pattern(pattern_pass_manager: _PatternPassManager):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormdynamicquant fusion will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip fusion.')
        return

    def search_fn(x1, x2, gamma, smooth_scales):
        y, _, xOut = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma)
        yOut, scale1Out = torch.ops.npu.npu_dynamic_quant.default(y, smooth_scales=smooth_scales)
        return yOut, xOut, scale1Out
    
    # For scenarios in DS networks, replace the operator combination of npu_add_rms_norm and npu_dynamic_quant 
    # with the npu_add_rms_norm_dynamic_quant operator. 
    # Since only the first path is quantized, pass [True, False] to the fusion operator's input parameter output_mask, 
    # indicating that only the first path is used for quantization.
    def replace_fn(x1, x2, gamma, smooth_scales):
        yOut, _, xOut, scale1Out, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(
            x1, x2, gamma,
            smooth_scale1=smooth_scales,
            output_mask=[True, False],
        )
        return yOut, xOut, scale1Out
    
    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 2), dtype=torch.float16)
        kwargs_tensor = functools.partial(torch.empty, 2, dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor(), kwargs_tensor(), kwargs_tensor())
        )


def _build_search_pattern() -> MultiOutputPattern:
    """
    Multi-output matching pattern equivalent to the operator combination in search_fn:
    
    def search_fn(x1, x2, gamma):
        y, _, xOut = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma)
        yOut, scale1Out = torch.ops.npu.npu_dynamic_quant.default(y.flatten(0, 1))
        scale1Out_view = scale1Out.view(-1, 1)
        return yOut, scale1Out_view, xOut
    
    - dynamic_quant_output0: 1st output of npu_dynamic_quant.default (corresponds to yOut in search_fn)
    - CallFunction(torch.ops.aten.view.default, dynamic_quant_output1, [-1, 1]): output of scale1Out.view(-1, 1) 
      (corresponds to scale1Out_view in search_fn)
    - add_rms_norm_output2: 3rd output of npu_add_rms_norm.default (corresponds to xOut in search_fn)
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
        0
    )

    add_rms_norm_output2 = CallFunction(
        operator.getitem,
        npu_add_rms_norm_func,
        2
    )

    npu_dynamic_quant_func = CallFunction(
        torch.ops.npu.npu_dynamic_quant.default, 
        CallFunction(
            torch.ops.aten.view.default,
            add_rms_norm_output0,
            Ignored()
        ),
        _users=2
    )

    dynamic_quant_output0 = CallFunction(
        operator.getitem,
        npu_dynamic_quant_func,
        0
    )

    dynamic_quant_output1 = CallFunction(
        operator.getitem,
        npu_dynamic_quant_func,
        1
    )

    return MultiOutputPattern([
            dynamic_quant_output0,
            CallFunction(
                torch.ops.aten.view.default,
                dynamic_quant_output1,
                [-1, 1]
            ),
            add_rms_norm_output2
        ])


@functools.lru_cache(None)
def _register_addrmsnormdynamicquant_pattern2(pattern_pass_manager: _PatternPassManager):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormdynamicquant fusion will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip fusion.')
        return

    def search_fn(x1, x2, gamma):
        pass
    
    def replace_fn(x1, x2, gamma):
        yOut, _, xOut, scale1Out, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant.default(
            x1, x2, gamma,
            output_mask=[True, False],
        )
        yOut_flatten = yOut.flatten(0, 1)
        scale1Out_view = scale1Out.view(-1, 1)
        return yOut_flatten, scale1Out_view, xOut

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
            extra_check=_check_view_shape,
            search_fn_pattern=_build_search_pattern()
        )