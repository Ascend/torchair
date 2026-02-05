__all__ = []

import functools
import sys
import torch

from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode

from torchair.core.utils import logger
from torchair.patterns.pattern_pass_manager import _PatternPassManager, _check_pattern_stream


def _pattern_extra_check(match: Match) -> bool:

    for node in match.nodes:
        if node.target == torch.ops.npu.npu_quantize.default:
            dtype_x0 = node.args[0].meta['val'].dtype
            # zero_points dtype only supports INT8 UINT8
            if node.args[2].meta['val'].dtype in [torch.uint8, torch.int8]:
                logger.debug("Parameter type mismatch in pattern match for npu_add_rms_norm_quant fusion pass.")
                return False
            # div_mode=False zero_points not supports FLOAT16
            if(len(node.args) == 6 and not node.args[5] and not (dtype_x0 == torch.bfloat16)):
                logger.debug("Parameter type mismatch in pattern match for npu_add_rms_norm_quant fusion pass.")
                return False
            # output dtype only supports INT8, QINT8
            if node.args[3] not in [1, 12]:  # 1: INT8, 12: QINT8
                logger.debug("Output type mismatch in pattern match for npu_add_rms_norm_quant fusion pass.")
                return False

        elif node.target == torch.ops.npu.npu_add_rms_norm.default:
            # div_mode=True only supports BFLOAT16 and FLOAT16
            x0_val = node.args[0].meta['val']
            x1_val = node.args[1].meta['val']
            if not ((x0_val.dtype == torch.float16 and x1_val.dtype == torch.float16) or
                    (x0_val.dtype == torch.bfloat16 and x1_val.dtype == torch.bfloat16)):
                logger.debug("Parameter type mismatch in pattern match for npu_add_rms_norm_quant fusion pass.")
                return False
            
            # last dim size should be multiple of 32 bytes
            if x1_val.ndim == 0:
                return False
            elem_bytes = x1_val.element_size()
            last_dim_elem = x1_val.shape[-1]
            last_dim_total_bytes = last_dim_elem * elem_bytes
            if not isinstance(last_dim_total_bytes, int) or last_dim_total_bytes % 32 != 0:
                logger.debug(
                    f"Input size detected in pattern match for npu_add_rms_norm_quant fusion pass."
                    f"Fusion is not supported because the input of the npu_add_rms_norm_quant must be divisible by 32 bytes."
                )
                return False

    return True


def _get_inputs():
    """
    generate example inputs for addrmsnormquant fusion
    """
    N, D = 2, 16
    xshape = [N, D]
    type_input = torch.float16  # BFLOAT16、FLOAT16
    gammashape = [D]

    x1 = torch.rand(xshape, dtype=type_input)
    x2 = torch.rand(xshape, dtype=type_input)
    gamma = torch.rand(gammashape, dtype=type_input)
    scales = torch.ones(gammashape, dtype=torch.float32)  # FLOAT32、BFLOAT16
    zero_points = torch.tensor(gammashape, dtype=torch.int32)  # INT32、BFLOAT16
    return (x1, x2, gamma, scales, zero_points)


@functools.lru_cache(None)
def _register_addrmsnormquant_pattern(pattern_pass_manager, div_mode):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormquant fusion will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip fusion.')
        return

    # replace the operator combination of npu_add_rms_norm and npu_quantize 
    # with the npu_add_rms_norm_quant operator. 
    def search_fn(x1, x2, gamma, scales, zero_points, epsilon, dtype):
        y, _, x_out = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma, epsilon)
        y_out = torch.ops.npu.npu_quantize.default(y, scales, zero_points=zero_points, dtype=dtype, axis=-1, div_mode=div_mode)
        return y_out, x_out

    def replace_fn(x1, x2, gamma, scales, zero_points, epsilon, _):
        # output dtype only supports INT8, axis currently only supports -1, passing other values will not take effect
        y1, _, x_out = torch.ops.npu.npu_add_rms_norm_quant.default(x1, x2, gamma, scales, zero_points, axis=-1, epsilon=epsilon)
        return y1, x_out

    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=_get_inputs(),
            extra_check=_pattern_extra_check,
            scalar_workaround={"epsilon": 2e-6, "dtype": 1},
            skip_duplicates=True
        )


@functools.lru_cache(None)
def _register_addrmsnormquant_pattern_default(pattern_pass_manager, div_mode):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The addrmsnormquant fusion will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip fusion.')
        return

    # replace the operator combination of npu_add_rms_norm and npu_quantize 
    # with the npu_add_rms_norm_quant operator. 
    def search_fn(x1, x2, gamma, scales, zero_points, dtype):
        y, _, x_out = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma)
        y_out = torch.ops.npu.npu_quantize.default(y, scales, zero_points=zero_points, dtype=dtype, axis=-1, div_mode=div_mode)
        return y_out, x_out

    def replace_fn(x1, x2, gamma, scales, zero_points, _):
        # output dtype only supports INT8, axis currently only supports -1, passing other values will not take effect
        y1, _, x_out = torch.ops.npu.npu_add_rms_norm_quant.default(x1, x2, gamma, scales, zero_points, axis=-1)
        return y1, x_out

    fake_mode = FakeTensorMode()
    with fake_mode:
        # sizes/values don't actually matter for initial trace
        # once we get a possible match we re-trace with the actual values and verify the match still holds
        pattern_pass_manager.register_pattern(
            search_fn=search_fn,
            replace_fn=replace_fn,
            example_inputs=_get_inputs(),
            extra_check=_pattern_extra_check,
            scalar_workaround={"dtype": 1},
            skip_duplicates=True
        )


@functools.lru_cache(None)
def _register_addrmsnormquant_patterns(pattern_pass_manager: _PatternPassManager):
    _register_addrmsnormquant_pattern_default(pattern_pass_manager, div_mode=True)
    _register_addrmsnormquant_pattern(pattern_pass_manager, div_mode=True)
    _register_addrmsnormquant_pattern_default(pattern_pass_manager, div_mode=False)
    _register_addrmsnormquant_pattern(pattern_pass_manager, div_mode=False)