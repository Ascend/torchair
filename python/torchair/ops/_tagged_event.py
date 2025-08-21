from typing import List
import threading
import torch
from torch.fx.node import has_side_effect
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.scope._scope_attr import _npu_get_or_create_stream_with_tag
from torchair.ge._ge_graph import Tensor, TensorSpec, ControlTensor, DataType
from ._lib import lib

try:
    import torch_npu
except ImportError as e:
    raise RuntimeError(
        "Couldn't import torch_npu. While use tagged_event_record/wait/ api, import torch_npu is necessary"
    ) from e

lib.define("tagged_event_record(str tag) -> ()")
lib.define("tagged_event_wait(str tag) -> ()")
lib.define("record_tagged_stream_(Tensor(a!) self, str tag) -> ()")
lib.define("record_tagged_stream(Tensor self, str tag) -> Tensor")

has_side_effect(torch.ops.air.tagged_event_record.default)
has_side_effect(torch.ops.air.tagged_event_wait.default)
has_side_effect(torch.ops.air.record_tagged_stream.default)
has_side_effect(torch.ops.air.record_tagged_stream_.default)

_GLOBAL_TAG_TO_EVENT = {}
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
    for tag, tagged_event in _GLOBAL_TAG_TO_EVENT.items():
        if tagged_event == event:
            return tag
    raise ValueError(f"tagged event {event} is not in use")


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


def _npu_record_tagged_stream(self: torch.Tensor, tagged_stream: str):
    return torch.ops.air.record_tagged_stream_(self, tagged_stream)


@torch.library.impl(lib, "tagged_event_record", "Meta")
def record_meta(tag: str):
    return None


@torch.library.impl(lib, "tagged_event_wait", "Meta")
def wait_meta(tag: str):
    return None


@torch.library.impl(lib, "record_tagged_stream_", "Meta")
def record_tagged_stream_inplace_meta(self: torch.Tensor, tagged_stream: str):
    return None


@torch.library.impl(lib, "record_tagged_stream", "Meta")
def record_tagged_stream_meta(self: torch.Tensor, tagged_stream: str):
    return self


@torch.library.impl(lib, "record_tagged_stream_", "Functionalize")
def record_tagged_stream_inplace_func(self: torch.Tensor, tagged_stream: str):
    self.copy_(torch.ops.air.record_tagged_stream(self, tagged_stream))


def record_impl(tag: str):
    event = _GLOBAL_TAG_TO_EVENT.get(tag)
    if event is None:
        raise AssertionError(f"tagged event is None while tag is {tag}, "
                             f"tagged event record failed, please check the tag!")
    logger.debug("tagged event record with tag = [%s]", tag)
    return event.record(torch.npu.current_stream())


def wait_impl(tag: str):
    event = _GLOBAL_TAG_TO_EVENT.get(tag)
    if event is None:
        raise AssertionError(f"tagged event is None while tag is {tag}, "
                             f"tagged event wait failed, please check the tag!")
    logger.debug("tagged event wait with tag = [%s]", tag)
    return event.wait(torch.npu.current_stream())


def record_tagged_stream_inplace_impl(self: torch.Tensor, tagged_stream: str):
    stream = _npu_get_or_create_stream_with_tag(tagged_stream)
    if stream is None:
        raise AssertionError(f"get stream with tag = {tagged_stream} failed")
    logger.debug(f"tagged stream = {stream} recorded with tag = {tagged_stream}")
    return self.record_stream(stream)


def record_tagged_stream_impl(self: torch.Tensor, tagged_stream: str):
    stream = _npu_get_or_create_stream_with_tag(tagged_stream)
    if stream is None:
        raise AssertionError(f"get stream with tag = {tagged_stream} failed")
    logger.debug(f"tagged stream = {stream} recorded with tag = {tagged_stream}")
    clone_tensor = self.clone()
    clone_tensor.record_stream(stream)
    return clone_tensor



torch.library.impl(lib, "tagged_event_record", "CompositeExplicitAutograd")(record_impl)
torch.library.impl(lib, "tagged_event_wait", "CompositeExplicitAutograd")(wait_impl)
torch.library.impl(lib, "record_tagged_stream", "CompositeExplicitAutograd")(record_tagged_stream_impl)
torch.library.impl(lib, "record_tagged_stream_", "CompositeExplicitAutograd")(record_tagged_stream_inplace_impl)


@register_fx_node_ge_converter(torch.ops.air.tagged_event_record.default)
def convert_event_record(tag: str,
                         meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.tagged_event_record.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.air.tagged_event_wait.default)
def convert_event_wait(tag: str,
                       meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.tagged_event_wait.default ge_converter is not implemented!")


@register_fx_node_ge_converter(torch.ops.air.record_tagged_stream.default)
def convert_record_tagged_stream(self: torch.Tensor,
                                 tagged_stream: str,
                                 meta_outputs: List[TensorSpec] = None):
    raise NotImplementedError("torch.ops.air.record_tagged_stream.default ge_converter is not implemented!")
