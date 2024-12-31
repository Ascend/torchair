import torch


def npu_print(*args, summarize_size=3):
    from ._print_ops import _npu_print
    return _npu_print(*args, summarize_size=summarize_size)


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor):
    from ._wait_tensor import _npu_wait_tensor
    return _npu_wait_tensor(self, dependency)