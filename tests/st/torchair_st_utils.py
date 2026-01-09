import sys
import copy
import io
import logging
import types
import warnings
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


@contextmanager
def capture_warnings():
    """
    Context manager to capture warnings using warnings.catch_warnings.

    Usage:
    with capture_warnings() as stdout:
        # code that prints to stdout
        print("Error message", file=sys.stdout)
    captured_output = stdout.getvalue()
    """
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # 捕获所有警告
        
        # 创建 StringIO 来存储格式化输出
        warnings_io = io.StringIO()
        
        yield warnings_io
        
        # 将捕获的警告列表格式化为字符串
        for warning in warning_list:
            warning_msg = f"{warning.category.__name__}: {warning.message}\n"
            warnings_io.write(warning_msg)


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


def create_reinplace_pass_wrapper(assert_func):
    """
    Create a wrapper for _reinplace_inplaceable_ops_pass to capture FX graphs before and after.
    
    Args:
        assert_func: Function that takes (graph_before, graph_after) and performs assertions.
                     graph_before and graph_after are torch.fx.GraphModule instances.
    
    Returns:
        A wrapper function that can be used to replace _reinplace_inplaceable_ops_pass.
    """
    # Import and save reference to original function at wrapper creation time
    from torchair._acl_concrete_graph.graph_pass import _reinplace_inplaceable_ops_pass
    original_func = _reinplace_inplaceable_ops_pass
    
    def wrapper(gm, multi_stream_enabled, *sample_args):
        # Save graph before reinplace
        graph_before = copy.deepcopy(gm)
        
        # Call original function
        ret = original_func(gm, multi_stream_enabled, *sample_args)
        
        # Save graph after reinplace
        graph_after = copy.deepcopy(gm)
        
        # Call assertion function
        assert_func(graph_before, graph_after)
        
        return ret
    
    return wrapper