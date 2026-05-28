# pylint: disable=R0401
# ruff: noqa: E402
import logging
import sys
import os
from enum import IntEnum

import torch
from inductor_npu_ext import config


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

class Soc(IntEnum):
    A2 = 1
    A3 = 2
    A5 = 3
    FUTURE = 999


def _detect_soc():
    """import 时探测当前 NPU SoC 版本。

    host-only 调试模式（cpu/nothrow）返回 None，不做 SoC gating。
    其余模式查 device name；torch_npu 未 load 或设备未就绪时不在 import 阶段
    硬崩，兜底为 None（等价"未知 SoC，不做 gating"）并告警。
    """
    if config._debugging_on_cpu:
        return None

    import torch_npu
    soc_name = torch.npu.get_device_name()
    if '910B' in soc_name:
        return Soc.A2
    if '910_93' in soc_name:
        return Soc.A3
    if '910_95' in soc_name or '950' in soc_name:
        return Soc.A5
    return Soc.FUTURE


current_soc = _detect_soc()

__all__ = [
    "FusedLayoutContractError",
    "logger",
    "maybe_check_fused_input_layout",
    "current_soc",
    "Soc",
]
