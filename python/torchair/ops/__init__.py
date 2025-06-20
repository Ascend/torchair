import torch


def npu_fused_infer_attention_score(*args, **kwargs):
    from ._npu_fused_infer_attention_score import _npu_fused_infer_attention_score
    return _npu_fused_infer_attention_score(*args, **kwargs)


def npu_print(*args, summarize_size=3):
    from ._print_ops import _npu_print
    return _npu_print(*args, summarize_size=summarize_size)


def npu_create_tagged_external_event(tag: str):
    from ._external_event import _npu_create_tagged_external_event
    return _npu_create_tagged_external_event(tag)


def npu_tagged_event_record(event):
    from ._external_event import _npu_tagged_event_record
    return _npu_tagged_event_record(event)


def npu_tagged_event_wait(event):
    from ._external_event import _npu_tagged_event_wait
    return _npu_tagged_event_wait(event)


def npu_tagged_event_reset(event):
    from ._external_event import _npu_tagged_event_reset
    return _npu_tagged_event_reset(event)