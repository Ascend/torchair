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
        self.stream_tag = stream_tag
        self.stream_priority = stream_priority

    def __enter__(self):
        from ._stream_in import _npu_stream_in
        return _npu_stream_in(self.stream_tag, self.stream_priority)

    def __exit__(self, *args):
        from ._stream_out import _npu_stream_out
        return _npu_stream_out(self.stream_tag, self.stream_priority)
