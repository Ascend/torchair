import logging
import multiprocessing
import os
import sys
from enum import IntEnum

import torch

from .. import config


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

class Soc(IntEnum):
    A2 = 1
    A3 = 2
    A5 = 3


def _detect_soc():
    """import 时探测当前 NPU SoC 版本。

    host-only 调试模式（cpu/nothrow）返回 None，不做 SoC gating。
    其余模式查 device name；torch_npu 未 load 或设备未就绪时不在 import 阶段
    硬崩，兜底为 None（等价"未知 SoC，不做 gating"）并告警。
    """
    if config._debugging_on_cpu:
        return None

    if multiprocessing.get_start_method(allow_none=True) == 'fork':
        return None

    import torch_npu
    soc_version = torch_npu.npu._backends.get_soc_version()
    if soc_version >= 260:
        return Soc.A5
    if soc_version >= 250:
        return Soc.A3
    if soc_version >= 220 and soc_version < 240:
        return Soc.A2
    logger.warning("Unknown SoC version %d name %s, SoC-specific check will be disabled",
                   soc_version, torch.npu.get_device_name())
    return None


current_soc = _detect_soc()

__all__ = [
    "logger",
    "current_soc",
    "Soc",
]
