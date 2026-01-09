__all__ = []

import functools
import sys
import torch

from torch._inductor.pattern_matcher import Match, CallFunction, KeywordArg, Ignored
from torch._subclasses.fake_tensor import FakeTensorMode

from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager, _check_pattern_stream


def _pattern_extra_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    Checks whether the K/N size of the npu_transpose_batchmatmul operation is divisible by 128
    """
    from torch._inductor.fx_passes.post_grad import same_meta

    def _check_input_tensor(node: torch.fx.Node) -> bool:
        kn = 2  # K/N
        for i, arg in enumerate(node.args):
            input_arg = arg.meta.get("val")
            if not isinstance(input_arg, torch.Tensor):
                return False

            if input_arg.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
                logger.debug(
                    f"input type detected in pattern match for npu_transpose_batchmatmul."
                    f"input{i} dtype: {input_arg.dtype}. "
                    f"Fusion is not supported because data types not in [torch.bfloat16, torch.float16, torch.float32]."
                )
                return False

            if input_arg.shape[kn] % 128 != 0:  # the K/N input of the npu_transpose_batchmatmul must be divisible by 128
                logger.debug(
                    f"input size detected in pattern match for npu_transpose_batchmatmul."
                    f"input{i} size: {input_arg.shape}. "
                    f"Fusion is not supported because the K/N input of the npu_transpose_batchmatmul must be divisible by 128."
                )
                return False

        return True

    if not _check_pattern_stream(match, "npu_transpose_batchmatmul"):
        return False

    for node in match.nodes:
        if node.target == torch.ops.aten.bmm.default:
            if not _check_input_tensor(node):
                return False
        if node.target == torch.ops.aten.view.default:
            input_node = node.args[0]
            output_node = node
            if not same_meta(input_node, output_node):
                logger.debug(
                    f"input and output size detected in pattern match for npu_transpose_batchmatmul."
                )
                return False

    return True


def _register_pattern_for_bmm_transpose(pattern_pass_manager: _PatternPassManager, build_search_pattern):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The batchmatmultranspose fusion will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip fusion.')
        return

    def search_fn(x1, x2):
        pass

    # For scenarios in long-cat networks, replace the operator
    # combination of btachmatmultranspose with the npu_transpose_batchmatmul operator
    def replace_fn(x1, x2):
        output = torch.ops.npu.npu_transpose_batchmatmul.default(x1, x2)
        return output

    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        input_tensor = functools.partial(torch.empty, (1, 1, 1), device="npu", dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor()),
            extra_check=_pattern_extra_check,
            search_fn_pattern=build_search_pattern()
        )


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern(pattern_pass_manager: _PatternPassManager, transpose_input1, transpose_input2):
    def _build_search_pattern():
        """
        The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
            y_bmm = torch.ops.aten.bmm.default(x1, x2)
            yOut= torch.ops.aten.transpose.int(y_bmm, 1, 0)
            return yOut
        """
        output = CallFunction(
            torch.ops.aten.transpose.int,
            CallFunction(
                torch.ops.aten.bmm.default,
                KeywordArg("x1"),
                KeywordArg("x2"),
            ),
            transpose_input1,
            transpose_input2
        )
        return output

    _register_pattern_for_bmm_transpose(pattern_pass_manager, _build_search_pattern)


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern1(pattern_pass_manager: _PatternPassManager, transpose_input1, transpose_input2):
    def _build_search_pattern():
        """
        The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
            y_bmm = torch.ops.aten.bmm.default(x1, x2)
            y_shape = torch.ops.aten.reshape.default(y_bmm, y_bmm.size)
            yOut= torch.ops.aten.transpose.int(y_shape, 1, 0)
            return yOut
        """
        bmm_output = CallFunction(
            torch.ops.aten.bmm.default,
            KeywordArg("x1"),
            KeywordArg("x2")
        )

        reshape_output = CallFunction(
            torch.ops.aten.view.default,
            bmm_output,
            Ignored()
        )

        output = CallFunction(
            torch.ops.aten.transpose.int,
            reshape_output,
            transpose_input1,
            transpose_input2
        )
        return output

    _register_pattern_for_bmm_transpose(pattern_pass_manager, _build_search_pattern)


@functools.lru_cache(None)
def _register_batchmatmultranspose_patterns(pattern_pass_manager: _PatternPassManager):
    _register_batchmatmultranspose_pattern(pattern_pass_manager, 0, 1)
    _register_batchmatmultranspose_pattern(pattern_pass_manager, 1, 0)
    _register_batchmatmultranspose_pattern1(pattern_pass_manager, 0, 1)
    _register_batchmatmultranspose_pattern1(pattern_pass_manager, 1, 0)