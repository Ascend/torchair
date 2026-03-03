__all__ = []

import functools
import sys
import torch

from torch._inductor.pattern_matcher import Match, CallFunction, KeywordArg, Ignored
from torch._subclasses.fake_tensor import FakeTensorMode

from npugraph_ex.core.utils import logger
from npugraph_ex.patterns.pattern_pass_manager import _PatternPassManager


def _pattern_extra_check(match: Match) -> bool:
    """
    Perform additional checks on the matched pattern.

    This function validates:
    1. K*B or K constraint based on perm_x1 configuration
    2. Input tensor dtype and K/N alignment
    3. View/Expand node input/output meta consistency (must be no-op)
    4. Transpose dimension arguments

    Args:
        match: The pattern match object.

    Returns:
        True if all checks pass, False otherwise.
    """
    from torch._inductor.fx_passes.post_grad import same_meta

    supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    kn_alignment = 128
    max_kb_value = 65536

    def _check_input_tensor(node: torch.fx.Node) -> bool:
        """Check if all input tensors of the bmm node meet the requirements."""
        kn_dim = 2  # K/N is the last dimension for 3D tensors
        for i, arg in enumerate(node.args):
            input_arg = arg.meta.get("val")
            if not isinstance(input_arg, torch.Tensor):
                return False

            # Check dtype
            if input_arg.dtype not in supported_dtypes:
                logger.debug(
                    f"Input type check failed for npu_transpose_batchmatmul. "
                    f"input{i} dtype: {input_arg.dtype}. "
                    f"Fusion is not supported because data types not in {supported_dtypes}."
                )
                return False

            # Check K/N alignment
            if input_arg.shape[kn_dim] % kn_alignment != 0:
                logger.debug(
                    f"K/N alignment check failed for npu_transpose_batchmatmul. "
                    f"input{i} size: {input_arg.shape}. "
                    f"Fusion is not supported because K/N must be divisible by {kn_alignment}."
                )
                return False

        return True

    def _check_kb_constraint() -> bool:
        """
        Check the K*B constraint based on perm_x1 configuration.

        - When perm_x1=[1, 0, 2] (input transpose), K*B must be in range [1, 65536)
        - When perm_x1=[0, 1, 2] (no input transpose), K must be less than 65536
        """
        # Count transpose nodes to determine pattern type
        transpose_count = sum(
            1 for node in match.nodes
            if node.target == torch.ops.aten.transpose.int
        )
        is_input_x1_transpose = transpose_count >= 2

        # Find the bmm node
        bmm_node = next(
            (n for n in match.nodes if n.target == torch.ops.aten.bmm.default),
            None
        )
        if bmm_node is None:
            logger.debug(
                f"BMM node not found in pattern match for npu_transpose_batchmatmul. "
                f"Fusion is not supported."
            )
            return False

        # Get the first input tensor (x1) shape: (B, M, K)
        x1_arg = bmm_node.args[0].meta.get("val")
        if x1_arg is None or not isinstance(x1_arg, torch.Tensor):
            logger.debug(
                f"Invalid x1 input for BMM node in pattern match for npu_transpose_batchmatmul. "
                f"x1_arg type: {type(x1_arg)}. Fusion is not supported."
            )
            return False

        b, k = x1_arg.shape[0], x1_arg.shape[2]

        if is_input_x1_transpose:
            # perm_x1=[1, 0, 2] case: K*B must be in range [1, 65536)
            kb_product = k * b
            if kb_product < 1 or kb_product >= max_kb_value:
                logger.debug(
                    f"K*B constraint check failed for npu_transpose_batchmatmul. "
                    f"K={k}, B={b}, K*B={kb_product}. "
                    f"Fusion is not supported because K*B must be in range [1, {max_kb_value}) "
                    f"when perm_x1=[1, 0, 2]."
                )
                return False
        else:
            # perm_x1=[0, 1, 2] case: K must be less than 65536
            if k >= max_kb_value:
                logger.debug(
                    f"K constraint check failed for npu_transpose_batchmatmul. "
                    f"K={k}. "
                    f"Fusion is not supported because K must be less than {max_kb_value} "
                    f"when perm_x1=[0, 1, 2]."
                )
                return False

        return True

    def _check_noop_input_output_meta(node: torch.fx.Node) -> bool:
        """Check if view/expand node's input and output have the same meta (must be no-op)."""
        input_node = node.args[0]
        output_node = node
        if not same_meta(input_node, output_node):
            logger.debug(
                f"{str(node.target)} node input/output meta mismatch for npu_transpose_batchmatmul. "
                f"Fusion is not supported because it must be a no-op."
            )
            return False
        return True

    def _check_transpose_args(node: torch.fx.Node) -> bool:
        """Check if the transpose node's dimension arguments are valid (only 0 or 1)."""
        dim1, dim2 = node.args[1], node.args[2]
        if not isinstance(dim1, int) or not isinstance(dim2, int):
            logger.debug(
                f"Transpose args must be int for npu_transpose_batchmatmul. "
                f"Got dim1={dim1} (type={type(dim1)}), dim2={dim2} (type={type(dim2)})."
            )
            return False
        if dim1 > 1 or dim2 > 1:
            logger.debug(
                f"Transpose args only support 0/1 for npu_transpose_batchmatmul. "
                f"Got dim1={dim1}, dim2={dim2}."
            )
            return False
        return True

    # Check K*B constraint
    if not _check_kb_constraint():
        return False

    # Check individual nodes
    for node in match.nodes:
        target = node.target
        if target == torch.ops.aten.bmm.default:
            if not _check_input_tensor(node):
                return False

        if target in (torch.ops.aten.view.default, torch.ops.aten.expand.default):
            if not _check_noop_input_output_meta(node):
                return False

        if target == torch.ops.aten.transpose.int:
            if not _check_transpose_args(node):
                return False

    return True


def _register_pattern_for_bmm_transpose(
        pattern_pass_manager: _PatternPassManager,
        build_search_pattern,
        input_x1: bool = False
):
    """
    Register a pattern for bmm + transpose fusion.

    Args:
        pattern_pass_manager: The pattern pass manager to register with.
        build_search_pattern: A callable that builds the search pattern.
        input_x1: If True, use perm_x1=[1, 0, 2]; otherwise use [0, 1, 2].
    """
    if 'torch_npu' not in sys.modules:
        logger.info(
            f"The batchmatmultranspose fusion will only be enabled in a torch npu env. "
            f"When there is no torch_npu in the env, skip fusion."
        )
        return

    def search_fn(x1, x2):
        pass

    perm_x1 = [1, 0, 2] if input_x1 else [0, 1, 2]

    def replace_fn(x1, x2):
        return torch.ops.npu.npu_transpose_batchmatmul.default(x1, x2, perm_x1=perm_x1)

    fake_mode = FakeTensorMode()
    with fake_mode:
        input_tensor = functools.partial(torch.empty, (1, 1, 1), dtype=torch.float16)
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=(input_tensor(), input_tensor()),
            extra_check=_pattern_extra_check,
            search_fn_pattern=build_search_pattern()
        )


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern(pattern_pass_manager: _PatternPassManager):
    """
    The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
        y_bmm = torch.ops.aten.bmm.default(x1, x2)
        yOut= torch.ops.aten.transpose.int(y_bmm, 1, 0)
        return yOut
    """
    def _build_search_pattern():
        return CallFunction(
            torch.ops.aten.transpose.int,
            CallFunction(
                torch.ops.aten.bmm.default,
                KeywordArg("x1"),
                KeywordArg("x2"),
            ),
            Ignored(),
            Ignored()
        )

    _register_pattern_for_bmm_transpose(pattern_pass_manager, _build_search_pattern)


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern1(pattern_pass_manager: _PatternPassManager):
    """
    The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
            y_bmm = torch.ops.aten.bmm.default(x1, x2)
            y_shape = torch.ops.aten.reshape.default(y_bmm, y_bmm.size)
            yOut= torch.ops.aten.transpose.int(y_shape, 1, 0)
            return yOut
    """
    def _build_search_pattern():
        bmm_output = CallFunction(
            torch.ops.aten.bmm.default,
            KeywordArg("x1"),
            KeywordArg("x2")
        )
        view_output = CallFunction(
            torch.ops.aten.view.default,
            bmm_output,
            Ignored()
        )
        return CallFunction(
            torch.ops.aten.transpose.int,
            view_output,
            Ignored(),
            Ignored()
        )

    _register_pattern_for_bmm_transpose(pattern_pass_manager, _build_search_pattern)


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern2(pattern_pass_manager: _PatternPassManager):
    """
    The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
            x1_t = torch.ops.aten.transpose.int(x1, dim1, dim2)
            x1_expand = torch.ops.aten.expand.default(x1_t, ...)
            x1_view = torch.ops.aten.view.default(x1_expand, ...)
            y_bmm = torch.ops.aten.bmm.default(x1_view, x2)
            y_view = torch.ops.aten.view.default(y_bmm, ...)
            y_out = torch.ops.aten.transpose.int(y_view, dim1, dim2)
            return y_out
    """
    def _build_search_pattern():
        transpose_output = CallFunction(
            torch.ops.aten.transpose.int,
            KeywordArg("x1"),
            Ignored(),
            Ignored()
        )
        expand_output = CallFunction(
            torch.ops.aten.expand.default,
            transpose_output,
            Ignored()
        )
        view_output = CallFunction(
            torch.ops.aten.view.default,
            expand_output,
            Ignored()
        )
        bmm_output = CallFunction(
            torch.ops.aten.bmm.default,
            view_output,
            KeywordArg("x2")
        )
        view_output2 = CallFunction(
            torch.ops.aten.view.default,
            bmm_output,
            Ignored()
        )
        return CallFunction(
            torch.ops.aten.transpose.int,
            view_output2,
            Ignored(),
            Ignored()
        )

    _register_pattern_for_bmm_transpose(
        pattern_pass_manager, _build_search_pattern, input_x1=True
    )


@functools.lru_cache(None)
def _register_batchmatmultranspose_pattern3(pattern_pass_manager: _PatternPassManager):
    """
    The search matching pattern equivalent to the operator combination in search_fn:

        def search_fn(x1, x2):
            x1_t = torch.ops.aten.transpose.int(x1, dim1, dim2)
            x1_expand = torch.ops.aten.expand.default(x1_t, ...)
            y_bmm = torch.ops.aten.bmm.default(x1_expand, x2)
            y_out = torch.ops.aten.transpose.int(y_bmm, dim1, dim2)
            return y_out
    """
    def _build_search_pattern():
        transpose_output = CallFunction(
            torch.ops.aten.transpose.int,
            KeywordArg("x1"),
            Ignored(),
            Ignored()
        )
        expand_output = CallFunction(
            torch.ops.aten.expand.default,
            transpose_output,
            Ignored()
        )
        bmm_output = CallFunction(
            torch.ops.aten.bmm.default,
            expand_output,
            KeywordArg("x2")
        )
        return CallFunction(
            torch.ops.aten.transpose.int,
            bmm_output,
            Ignored(),
            Ignored()
        )

    _register_pattern_for_bmm_transpose(
        pattern_pass_manager, _build_search_pattern, input_x1=True
    )


def _register_batchmatmultranspose_patterns(pattern_pass_manager: _PatternPassManager):
    """
    Register all batchmatmul + transpose fusion patterns.

    Fuses bmm + transpose operator combinations into npu_transpose_batchmatmul
    for better performance. Supported patterns:

        - pattern:  bmm(x1, x2) -> transpose                              (perm_x1=[0,1,2])
        - pattern1: bmm(x1, x2) -> view -> transpose                      (perm_x1=[0,1,2])
        - pattern2: transpose(x1) -> expand -> view -> bmm -> view -> transpose  (perm_x1=[1,0,2])
        - pattern3: transpose(x1) -> expand -> bmm -> transpose           (perm_x1=[1,0,2])

    Patterns are registered in a specific order to ensure proper matching priority:
    more specific patterns (pattern2, pattern3) are registered first, then pattern
    and pattern1.
    """
    _register_batchmatmultranspose_pattern2(pattern_pass_manager)
    _register_batchmatmultranspose_pattern3(pattern_pass_manager)
    _register_batchmatmultranspose_pattern(pattern_pass_manager)
    _register_batchmatmultranspose_pattern1(pattern_pass_manager)