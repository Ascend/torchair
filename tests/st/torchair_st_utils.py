import sys
import io
import types
from contextlib import contextmanager


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
    try:
        sys.stdout = io.StringIO()
        yield sys.stdout
    finally:
        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        # Optionally print the captured output if you want to see it
        print("Captured stdout message:\n", captured_output, file=old_stdout)


def generate_faked_module():
    def is_available():
        return True

    # create a new module to fake torch.npu dynamicaly
    npu = types.ModuleType("npu")

    npu.is_available = is_available

    return npu
