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
