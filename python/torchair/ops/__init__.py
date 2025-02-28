import torch


def npu_print(*args, summarize_size=3):
    from ._print_ops import _npu_print
    return _npu_print(*args, summarize_size=summarize_size)


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor):
    from ._wait_tensor import _npu_wait_tensor
    return _npu_wait_tensor(self, dependency)


class NpuStreamSwitch:
    def __init__(self, stream_tag: str, stream_priority: int = 0):
        self.keys = ["_user_stream_label", "_user_stream_priority"]
        self.values = [stream_tag, str(stream_priority)]

    def __enter__(self):
        from ._scope_enter import _npu_scope_enter
        return _npu_scope_enter(self.keys, self.values)

    def __exit__(self, *args):
        from ._scope_exit import _npu_scope_exit
        return _npu_scope_exit()


class SuperKernelScope:
    def __init__(self, scope: str, options: str = None):
        self.keys = ["_super_kernel_scope", "_super_kernel_options"]
        self.values = [scope, options]

    def __enter__(self):
        from ._scope_enter import _npu_scope_enter
        return _npu_scope_enter(self.keys, self.values)

    def __exit__(self, *args):
        from ._scope_exit import _npu_scope_exit
        return _npu_scope_exit()
