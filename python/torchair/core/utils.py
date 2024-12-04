from functools import lru_cache
import logging
import sys
import os
import threading
from datetime import datetime

__all__ = []


class _MillisecAndMicrosecFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        s = ct.strftime("%Y-%m-%d %H:%M:%S")
        s += ('.{:03d}'.format(ct.microsecond // 1000) + \
              '.{:03d}'.format(ct.microsecond % 1000))
        return s


def _get_logger(*, level=logging.ERROR, output=sys.stdout, file=None, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = _MillisecAndMicrosecFormatter(
        f'[%(levelname)s] TORCHAIR(%(process)d,{os.path.basename(sys.executable)})' \
        f':%(asctime)s [%(filename)s:%(lineno)d]{threading.get_native_id()} %(message)s')

    if output:
        console_handler = logging.StreamHandler(output)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    @lru_cache
    def _warning_once(msg):
        logger.warning(msg)

    logger.warning_once = _warning_once

    return logger


logger = _get_logger(name="torchair")
