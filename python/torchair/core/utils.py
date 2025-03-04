__all__ = []

from functools import lru_cache
import logging
import sys
import os
import threading
from datetime import datetime


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
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        torchair_logger.addHandler(file_handler)

    @lru_cache
    def _warning_once(msg):
        torchair_logger.warning(msg)

    torchair_logger.warning_once = _warning_once

    return torchair_logger


logger = _get_logger(name="torchair")
