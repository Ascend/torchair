import torch


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor):
    from ._wait_tensor import _npu_wait_tensor
    return _npu_wait_tensor(self, dependency)


class _Scope:
    def __init__(self, attrs):
        self.attrs = attrs

    def __enter__(self):
        from ._scope import _npu_scope_enter
        return _npu_scope_enter(self.attrs)

    def __exit__(self, *args):
        from ._scope import _npu_scope_exit
        return _npu_scope_exit()


def npu_stream_switch(stream_tag: str, stream_priority: int = 0):
    return _Scope([
        ("_user_stream_label", stream_tag),
        ("_user_stream_priority", str(stream_priority))
    ])


def super_kernel(scope: str, options: str = ''):
    return _Scope([
        ("_super_kernel_scope", scope),
        ("_super_kernel_options", options)
    ])


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    return _Scope([
        ("_op_aicore_num", str(op_aicore_num)),
        ("_op_vectorcore_num", str(op_vectorcore_num))
    ])