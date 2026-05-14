# pylint: disable=W1203,R0401
from __future__ import annotations

import os
from typing import Any, Sequence

import torch

from inductor_npu_ext.config import _check_layout_enabled
from inductor_npu_ext.common import logger


class FusedLayoutContractError(RuntimeError):
    """Fused kernel input tensor shape/stride/dtype/device mismatch."""


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _safe_int_convert(val) -> int:
    """Safely convert SymInt/sympy.Expr/int to integer."""
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError, OverflowError):
        raise FusedLayoutContractError(f"Cannot convert to int: {val} (type={type(val).__name__})")


def maybe_check_fused_input_layout(
    *,
    kernel_name: str,
    buffer_name: str,
    tensor: torch.Tensor,
    expected_sizes: Sequence[Any],
    expected_strides: Sequence[Any],
    expected_dtype: torch.dtype,
    expected_device_type: str = "npu",
    path=None,
) -> None:
    debug_log = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1"
    if not _check_layout_enabled and not debug_log:
        return

    act_sizes = tuple(_safe_int_convert(t) for t in tensor.size())
    act_strides = tuple(_safe_int_convert(t) for t in tensor.stride())
    exp_sizes = tuple(_safe_int_convert(t) for t in expected_sizes)
    exp_strides = tuple(_safe_int_convert(t) for t in expected_strides)

    # Check dimension count first
    if len(act_sizes) != len(exp_sizes):
        msg = (
            f"ndim mismatch kernel={kernel_name!r} buffer={buffer_name!r}: expected {len(exp_sizes)}, got {len(act_sizes)}, "
            f"graph path={path!r}"
        )
        if _check_layout_enabled:
            raise FusedLayoutContractError(msg)
        logger.warning(msg)
        return

    # Empty tensor: skip layout checks (matches Inductor assert_size_stride logic)
    # "comparing strides for 0 size tensor is tricky. Ignore them for now."
    if tensor.numel() == 0:
        if debug_log:
            logger.info(f"kernel {kernel_name!r}: buffer {buffer_name!r} empty tensor, skip layout check")
        return

    # Collect mismatch reasons
    mismatches = []

    if act_sizes != exp_sizes:
        dims = [
            f"dim{i}: {act_sizes[i]} != {exp_sizes[i]}" for i in range(len(act_sizes)) if act_sizes[i] != exp_sizes[i]
        ]
        mismatches.append(f"size: {', '.join(dims)}")

    stride_dims = [
        f"dim{i}: stride {act_strides[i]} != {exp_strides[i]} (size={act_sizes[i]})"
        for i in range(len(act_sizes))
        if act_sizes[i] > 1 and act_strides[i] != exp_strides[i]
    ]
    if stride_dims:
        mismatches.append(f"stride: {', '.join(stride_dims)}")

    if tensor.dtype != expected_dtype:
        mismatches.append(f"dtype: {_dtype_name(tensor.dtype)} != {_dtype_name(expected_dtype)}")

    if tensor.device.type != expected_device_type:
        mismatches.append(f"device: {tensor.device.type} != {expected_device_type}")

    if not mismatches:
        if debug_log:
            logger.info(f"kernel {kernel_name!r}: buffer {buffer_name!r} layout check ok")
        return

    msg = (
        f"layout mismatch kernel={kernel_name!r} buffer={buffer_name!r}: {'; '.join(mismatches)}. "
        f"expected size={exp_sizes} stride={exp_strides} dtype={_dtype_name(expected_dtype)}, "
        f"actual size={act_sizes} stride={act_strides} dtype={_dtype_name(tensor.dtype)}, "
        f"graph path={path!r})"
    )
    if _check_layout_enabled:
        raise FusedLayoutContractError(msg)
    logger.warning(msg)


IMPORT_LINE = "from inductor_npu_ext.common.fused_layout_check import maybe_check_fused_input_layout"
