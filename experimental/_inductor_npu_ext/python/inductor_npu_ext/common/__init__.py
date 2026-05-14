# pylint: disable=R0401
# ruff: noqa: E402
import logging
import sys
import os


def _setup_logger(name):
    """Setup logger with RANK prefix."""
    rank = os.getenv("RANK", "0")
    level = logging.DEBUG if os.getenv("TORCH_COMPILE_DEBUG", "0") == "1" else logging.ERROR

    default_logger = logging.getLogger(name)
    default_logger.setLevel(level)

    if not default_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        format_str = f"[%(asctime)s] [RANK{rank}] [%(levelname)s] %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        default_logger.addHandler(handler)
        default_logger.propagate = False

    return default_logger


logger = _setup_logger("inductor_asc")

from inductor_npu_ext.common.fused_layout_check import (
    FusedLayoutContractError,
    maybe_check_fused_input_layout,
)

__all__ = [
    "FusedLayoutContractError",
    "logger",
    "maybe_check_fused_input_layout",
]
