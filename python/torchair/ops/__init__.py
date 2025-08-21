import torch


def npu_fused_infer_attention_score(*args, **kwargs):
    from ._npu_fused_infer_attention_score import _npu_fused_infer_attention_score
    return _npu_fused_infer_attention_score(*args, **kwargs)

def npu_fused_infer_attention_score_v2(*args, **kwargs):
    from ._npu_fused_infer_attention_score_v2 import _npu_fused_infer_attention_score_v2
    return _npu_fused_infer_attention_score_v2(*args, **kwargs)

def npu_print(*args, summarize_size=3):
    from ._print_ops import _npu_print
    return _npu_print(*args, summarize_size=summarize_size)


def npu_create_tagged_event(tag: str):
    from ._tagged_event import _npu_create_tagged_event
    return _npu_create_tagged_event(tag)


def npu_tagged_event_record(event):
    from ._tagged_event import _npu_tagged_event_record
    return _npu_tagged_event_record(event)


def npu_tagged_event_wait(event):
    from ._tagged_event import _npu_tagged_event_wait
    return _npu_tagged_event_wait(event)


# this api is used to record tensor to tagged stream with torch.compile(mode='reduce-overhead') with npu backend,
# it will create a new stream with tag if not exist, if you are in main stream, please use tagged_stream = 'default'
def npu_record_tagged_stream(self: torch.Tensor, tagged_stream: str):
    from ._tagged_event import _npu_record_tagged_stream
    return _npu_record_tagged_stream(self, tagged_stream)