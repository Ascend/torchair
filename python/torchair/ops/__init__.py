from ._print_ops import _npu_print


def npu_print(*args, summarize_size=3):
    return _npu_print(*args, summarize_size=summarize_size)
