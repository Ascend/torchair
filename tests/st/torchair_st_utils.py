import sys
import io
import logging
import types
from contextlib import contextmanager

import torch


@contextmanager
def capture_stdout():
    """
    Context manager to capture stdout output.

    Usage:
    with capture_stdout() as stdout:
        # code that prints to stdout
        print("Error message", file=sys.stdout)
    captured_output = stdout.getvalue()
    """

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield sys.stdout
    finally:
        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        # Optionally print the captured output if you want to see it
        print("Captured stdout message:\n", captured_output, file=old_stdout)


@contextmanager
def capture_logger():
    """
    Context manager to capture python logger output.

    Usage:
    with capture_logger() as stdout:
        # code that prints to stdout by logger
    captured_output = stdout.getvalue()
    """

    capture_logger = logging.getLogger()
    stream_io = io.StringIO()
    handler = logging.StreamHandler(stream_io)
    capture_logger.addHandler(handler)

    try:
        yield stream_io
    finally:
        captured_output = stream_io.getvalue()
        capture_logger.removeHandler(handler)

        # Optionally print the captured output if you want to see it
        print("Captured logger message:\n", captured_output)


def generate_faked_module():
    def is_available():
        return True

    # create a new module to fake torch.npu dynamicaly
    npu = types.ModuleType("npu")

    npu.is_available = is_available

    return npu


def register_is_npu():
    @property
    def _is_npu(self):
        return not self.is_cpu

    torch.Tensor.is_npu = _is_npu