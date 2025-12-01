import torch
from ._scope import _npu_scope_enter, _npu_scope_exit


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor):
    from ._wait_tensor import _npu_wait_tensor
    return _npu_wait_tensor(self, dependency)


class _Scope:
    def __init__(self, attrs):
        self.attrs = attrs

    def __enter__(self):
        return _npu_scope_enter(self.attrs)

    def __exit__(self, *args):
        return _npu_scope_exit()


def npu_stream_switch(stream_tag: str, stream_priority: int = 0):
    return _Scope([
        ("_user_stream_label", stream_tag),
        ("_user_stream_priority", str(stream_priority))
    ])


def super_kernel(scope: str, options: str = ''):
    if scope is None:
        scope = ''

    return _Scope([
        ("_super_kernel_scope", scope),
        ("_super_kernel_options", options)
    ])


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    return _Scope([
        ("_op_aicore_num", str(op_aicore_num)),
        ("_op_vectorcore_num", str(op_vectorcore_num))
    ])


def op_never_timeout(enable: bool = True):
    """
    Adds the '_op_exec_never_timeout' attribute to an operator. This function returns a scope object 
    that configures the operator's timeout behavior.

    Args:
    enable (bool): Specifies whether to enable the '_op_exec_never_timeout' attribute. Default is True, 
                   meaning the operator will never timeout.

    Returns:
    _Scope: A scope object containing the configuration for the timeout attribute. This object can 
            be used within a context manager to apply the '_op_exec_never_timeout' attribute to specific operations.

    Example usage:
    with op_never_timeout(True):
        # Perform operations that will never timeout
    
    """
    if not isinstance(enable, bool):
        raise TypeError(
            f"op_never_timeout() argument 'enable' must be bool, but got {type(enable).__name__}."
        )

    return _Scope([
        ("_op_exec_never_timeout", str(enable))
    ])


def profiler_trace(index: str, mode: str):
    """
    Context manager that tags ops in its scope with profiling attributes.
    
    Behavior summary:
    - If the scope contains exactly one op:
        that op is tagged with the attribute that matches the mode ("begin", "end", or "both").
    - If the scope contains more than one op:
        mode == "both": the first op is tagged "begin" and the last op is tagged "end".
        mode == "begin": the first op is tagged "begin".
        mode == "end": the last op is tagged "end".
    
    Args:
        index (str): Unique profiling ID.
        mode (str): Tagging mode — "begin", "end", or "both".
    
    Supported scenarios and semantics:
    1. mode == "begin"
        Example:
            sub1 = torch.sub(in3, in4)
            with torchair.scope.profiler_trace('index1', 'begin'):
                add1 = torch.add(in3, in4)
                cat1 = torch.cat([in1, in4], dim=1)
                mm1 = torch.mm(in3, in4)
            add2 = torch.add(in3, in4)
            mm2 = torch.mm(in3, in4)
        Result:
            add1 is tagged "begin", a timestamp will be marked begin add1.
    
    2. mode == "end"
        Example:
            sub1 = torch.sub(in3, in4)
            with torchair.scope.profiler_trace('index1', 'end'):
                add1 = torch.add(in3, in4)
                cat1 = torch.cat([in1, in4], dim=1)
                mm1 = torch.mm(in3, in4)
            add2 = torch.add(in3, in4)
            mm2 = torch.mm(in3, in4)
        Result:
            mm1 is tagged "end", a timestamp will be marked end mm1.
    
    3. mode == "both" with a single op in scope
        Example:
            sub1 = torch.sub(in3, in4)
            with torchair.scope.profiler_trace('index1', 'both'):
                add1 = torch.add(in3, in4)
            add2 = torch.add(in3, in4)
            mm2 = torch.mm(in3, in4)
        Result:
            add1 is tagged "both", a timestamp will be marked begin add1 and another end add1
    
    4. mode == "both" with multiple ops in scope
        Example:
            sub1 = torch.sub(in3, in4)
            with torchair.scope.profiler_trace('index1', 'both'):
                add1 = torch.add(in3, in4)
                cat1 = torch.cat([in1, in4], dim=1)
                mm1 = torch.mm(in3, in4)
            add2 = torch.add(in3, in4)
            mm2 = torch.mm(in3, in4)
        Result:
            add1 is tagged "begin" and mm1 is tagged "end", a timestamp will be marked begin add1,
            and another timestamp will be marked end mm1.
    
    Unsupported scenarios:
    - Nested profiler_trace contexts are not supported. Example of unsupported nesting:
          with torchair.scope.profiler_trace('index1', 'end'):
              add1 = torch.add(in3, in4)
              with torchair.scope.profiler_trace('index1', 'begin'):
                  mm1 = torch.mm(in3, in4)
    - If nested contexts are detected, a ValueError is raised:
          raise ValueError(f"Nested profiler_trace is not allowed")
    """
    return _Scope([
        ("_profiler_trace_index", index),
        ("_profiler_trace_pos", mode)
    ])
