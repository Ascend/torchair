from typing import List
import threading
import torch
from torch.fx.node import has_side_effect
from packaging import version
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.scope._scope_attr import _npu_get_or_create_stream_with_tag
from torchair.ge._ge_graph import Tensor, TensorSpec, ControlTensor, DataType
from ._lib import lib
from ._utils import TaggedEventBidict, TaggedEventFakeBidict

try:
    import torch_npu
except ImportError as e:
    raise RuntimeError(
        "Couldn't import torch_npu. While use tagged_event_record/wait/ api, import torch_npu is necessary"
    ) from e

lib.define("tagged_event_record(str tag, bool created_inside=False) -> ()")
lib.define("tagged_event_wait(str tag, bool created_inside=False) -> ()")
lib.define("record_tagged_stream_(Tensor(a!) input, str tag) -> ()")
lib.define("record_tagged_stream(Tensor input, str tag) -> ()")

has_side_effect(torch.ops.air.tagged_event_record.default)
has_side_effect(torch.ops.air.tagged_event_wait.default)
has_side_effect(torch.ops.air.record_tagged_stream.default)

# 2.5及以下版本dynamo原生不支持built-in里面的call_id操作，导致fullgraph成图时触发unsupported报错。仅在2.6及以上版本使用id实现双向字典提升event查找效率。
if version.parse(torch.__version__) >= version.parse("2.6.0"):
    _GLOBAL_TAG_TO_EVENT = TaggedEventBidict()
else:
    _GLOBAL_TAG_TO_EVENT = TaggedEventFakeBidict()
_GLOBAL_LOCK = threading.Lock()


def _npu_create_tagged_event(tag: str):
    with _GLOBAL_LOCK:
        if tag in _GLOBAL_TAG_TO_EVENT.keys():
            raise ValueError(f"tagged event tag {tag} is already in use")
        tagged_event = torch.npu.Event()
        _GLOBAL_TAG_TO_EVENT[tag] = tagged_event
        logger.debug("create tagged event with tag = [%s] successfully", tag)
        return tagged_event


def _get_tag_by_event(event: torch.npu.Event) -> str:
    tag = _GLOBAL_TAG_TO_EVENT.get_reverse(event)
    if tag is None:
        raise ValueError(f"tagged event {event} is not in use")
    return tag


def _npu_tagged_event_record(event: torch.npu.Event):
    tag = _get_tag_by_event(event)
    if tag is None:
        raise AssertionError(
            "call npu_tagged_event_record failed while event tag is None, please use "
            "torchair.ops.npu_create_tagged_tagged_event(tag: str) to create event then do event record!")
    return torch.ops.air.tagged_event_record(tag)


def _npu_tagged_event_wait(event: torch.npu.Event):
    tag = _get_tag_by_event(event)
    if tag is None:
        raise AssertionError(
            "call npu_tagged_event_wait failed while event tag is None, please use "
            "torchair.ops.npu_create_tagged_tagged_event(tag: str) to create event then do event wait!")
    return torch.ops.air.tagged_event_wait(tag)


def _npu_record_tagged_stream(input: torch.Tensor, tagged_stream: str):
    return torch.ops.air.record_tagged_stream_(input, tagged_stream)


@torch.library.impl(lib, "tagged_event_record", "Meta")
def record_meta(tag: str, created_inside: bool = False):
    return None


@torch.library.impl(lib, "tagged_event_wait", "Meta")
def wait_meta(tag: str, created_inside: bool = False):
    return None


@torch.library.impl(lib, "record_tagged_stream_", "Meta")
def record_tagged_stream_inplace_meta(input: torch.Tensor, tagged_stream: str):
    return None


@torch.library.impl(lib, "record_tagged_stream", "Meta")
def record_tagged_stream_meta(input: torch.Tensor, tagged_stream: str):
    return None


@torch.library.impl(lib, "record_tagged_stream_", "Functionalize")
def record_tagged_stream_inplace_func(input: torch.Tensor, tagged_stream: str):
    # The record_stream interface does not involve input mutation,
    # so there is no need to copy the output of out-of-place op to the original input.
    torch.ops.air.record_tagged_stream(input, tagged_stream)


def _get_event_by_tag(tag: str, created_inside: bool):
    if not created_inside:
        event = _GLOBAL_TAG_TO_EVENT.get(tag)
    else:
        from torchair._acl_concrete_graph.graph_pass import _GLOBAL_SCOPE_TAG_TO_EVENT
        event = _GLOBAL_SCOPE_TAG_TO_EVENT.get(tag)
    return event


def record_impl(tag: str, created_inside: bool = False):
    event = _get_event_by_tag(tag, created_inside)
    if event is None:
        custom_warning = 'please report an issue to Torchair!'
        inside_warning = f'please make sure you have created tag {tag} with npu_create_tagged_event API.'
        raise AssertionError(f"tagged event is None while tag is {tag}, "
                             f"tagged event record failed, "
                             f"{inside_warning if created_inside else custom_warning}")
    logger.debug("tagged event record with tag = [%s]", tag)
    return event.record(torch.npu.current_stream())


def wait_impl(tag: str, created_inside: bool = False):
    event = _get_event_by_tag(tag, created_inside)
    if event is None:
        custom_warning = 'please report an issue to Torchair!'
        inside_warning = f'please make sure you have created tag {tag} with npu_create_tagged_event API.'
        raise AssertionError(f"tagged event is None while tag is {tag}, "
                             f"tagged event wait failed, "
                             f"{inside_warning if created_inside else custom_warning}")
    logger.debug("tagged event wait with tag = [%s]", tag)
    return event.wait(torch.npu.current_stream())


def record_tagged_stream_impl(input: torch.Tensor, tagged_stream: str):
    stream = _npu_get_or_create_stream_with_tag(tagged_stream)
    if stream is None:
        raise AssertionError(f"get stream with tag = {tagged_stream} failed")
    logger.debug(f"tagged stream = {stream} recorded with tag = {tagged_stream}")
    # The record_stream interface in PyTorch does not directly modify the input,
    # it obtains the data_ptr of the input and increases the stream use count.
    # Therefore, if we first add clone for input,
    # this will cause the recorded data_ptr used for multi-stream usage to become the data_ptr of the cloned tensor.
    input.record_stream(stream)


torch.library.impl(lib, "tagged_event_record", "CompositeExplicitAutograd")(record_impl)
torch.library.impl(lib, "tagged_event_wait", "CompositeExplicitAutograd")(wait_impl)
torch.library.impl(lib, "record_tagged_stream", "CompositeExplicitAutograd")(record_tagged_stream_impl)
torch.library.impl(lib, "record_tagged_stream_", "CompositeExplicitAutograd")(record_tagged_stream_impl)


@register_fx_node_ge_converter(torch.ops.air.tagged_event_record.default)
def convert_event_record(tag: str,
                         created_inside: bool = False,
                         meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.tagged_event_record.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.air.tagged_event_wait.default)
def convert_event_wait(tag: str,
                       created_inside: bool = False,
                       meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.tagged_event_wait.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.air.record_tagged_stream.default)
def convert_record_tagged_stream(input: torch.Tensor,
                                 tagged_stream: str,
                                 meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.record_tagged_stream.default ge_converter is not implemented!")