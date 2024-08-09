from ._print_ops import _npu_print


def npu_print(*args, summarize_num=3):
    return _npu_print(*args, summarize_num=summarize_num)
