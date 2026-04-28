__all__ = []

from functools import lru_cache
import logging
import sys
import os
import threading
from datetime import datetime


def _create_debug_log_paths():
    if os.getenv("TORCH_COMPILE_DEBUG") != "1":
        return None

    from torch._dynamo.utils import get_debug_dir
    base_dir = os.path.join(get_debug_dir(), "torchair")
    os.makedirs(base_dir, exist_ok=True)
    debug_log_path = os.path.join(base_dir, "debug.log")

    from torchair.core import _torchair
    _torchair.SetDebugLogPath(debug_log_path)

    return debug_log_path

_torchair_debug_log_path = None
_TORCHAIR_RUN_NAME_PATCHED = False


def _init_debug_logging():
    global _torchair_debug_log_path
    if _torchair_debug_log_path:
        return _torchair_debug_log_path
    path = _create_debug_log_paths()
    if not path:
        return None
    formatter = _MillisecAndMicrosecFormatter(
        f'[%(levelname)s] TORCHAIR({os.getpid()},{os.path.basename(sys.executable)})'
        f':%(asctime)s [%(filename)s:%(lineno)d]{threading.get_native_id()} %(message)s')
    fh = logging.FileHandler(path, encoding='utf-8', delay=False)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    _torchair_debug_log_path = path
    return path


def _set_dynamo_run_dir_name():
    if os.getenv("TORCH_COMPILE_DEBUG") != "1":
        return
    global _TORCHAIR_RUN_NAME_PATCHED
    try:
        from torch._dynamo import utils as dutils
    except ImportError as e:
        warnings.warn(f"set_dynamo_run_dir_name skipped: cannot import torch._dynamo.utils: {e}")
        return
    if _TORCHAIR_RUN_NAME_PATCHED:
        return
    orig = dutils.get_debug_dir

    def ranked_get_debug_dir():
        base = orig()
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                return base
            rank_id = dist.get_rank()
        except (ImportError, AttributeError):
            return base

        parent, name = os.path.dirname(base), os.path.basename(base)
        suffix = f"rank_{rank_id}"
        if not name.endswith(suffix):
            name = f"{name}-{suffix}"
        return os.path.join(parent, name)

    dutils.get_debug_dir = ranked_get_debug_dir
    _TORCHAIR_RUN_NAME_PATCHED = True

EVENT_LEVEL = 35
logging.addLevelName(EVENT_LEVEL, 'EVENT')


class _EventLogger(logging.Logger):
    def event(self, msg, *args, **kwargs):
        if self.isEnabledFor(EVENT_LEVEL):
            self._log(EVENT_LEVEL, msg, args, **kwargs)


logging.setLoggerClass(_EventLogger)


class _MillisecAndMicrosecFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        s = ct.strftime("%Y-%m-%d %H:%M:%S")
        s += ('.{:03d}'.format(ct.microsecond // 1000) + \
              '.{:03d}'.format(ct.microsecond % 1000))
        return s


def _get_logger(*, level=logging.ERROR, output=sys.stdout, file=None, name=None):
    torchair_logger = logging.getLogger(name)
    torchair_logger.setLevel(level)

    formatter = _MillisecAndMicrosecFormatter(
        f'[%(levelname)s] TORCHAIR({os.getpid()},{os.path.basename(sys.executable)})' \
        f':%(asctime)s [%(filename)s:%(lineno)d]{threading.get_native_id()} %(message)s')

    if output:
        console_handler = logging.StreamHandler(output)
        console_handler.setFormatter(formatter)
        torchair_logger.addHandler(console_handler)

    if file:
        file_handler = logging.FileHandler(file, encoding='utf-8', delay=False)
        file_handler.setFormatter(formatter)
        torchair_logger.addHandler(file_handler)

    @lru_cache
    def _warning_once(msg):
        torchair_logger.warning(msg)

    torchair_logger.warning_once = _warning_once

    return torchair_logger

logger = _get_logger(
    name="torchair",
    file=None,
    level=logging.DEBUG if os.getenv("TORCH_COMPILE_DEBUG") == "1" else logging.ERROR
)
