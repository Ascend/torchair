from typing import List
import threading
import torch
from torch.fx.node import has_side_effect
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, ControlTensor, DataType
from ._lib import lib

try:
    import torch_npu
except ImportError as e:
    raise RuntimeError(
        "Couldn't import torch_npu. While use external_event_record/wait/reset/ api, import torch_npu is necessary"
    ) from e

lib.define("external_event_record(str tag) -> ()")
lib.define("external_event_wait(str tag) -> ()")
lib.define("external_event_reset(str tag) -> ()")

has_side_effect(torch.ops.air.external_event_record.default)
has_side_effect(torch.ops.air.external_event_wait.default)
has_side_effect(torch.ops.air.external_event_reset.default)

_GLOBAL_TAG_TO_EVENT = {}
_GLOBAL_LOCK = threading.Lock()


def _npu_create_tagged_external_event(tag: str):
    with _GLOBAL_LOCK:
        if tag in _GLOBAL_TAG_TO_EVENT.keys():
            raise ValueError(f"external event tag {tag} is already in use")
        external_event = torch.npu.ExternalEvent()
        _GLOBAL_TAG_TO_EVENT[tag] = external_event
        logger.debug("create external event with tag = [%s] successfully", tag)
        return external_event


def _get_tag_by_event(event: torch.npu.ExternalEvent) -> str:
    for tag, external_event in _GLOBAL_TAG_TO_EVENT.items():
        if external_event == event:
            return tag
    raise ValueError(f"external event {event} is not in use")


def _npu_tagged_event_record(event: torch.npu.ExternalEvent):
    tag = _get_tag_by_event(event)
    if tag is None:
        raise AssertionError(
            "call npu_tagged_event_record failed while event tag is None, please use "
            "torchair.ops.npu_create_tagged_external_event(tag: str) to create event then do event record!")
    return torch.ops.air.external_event_record(tag)


def _npu_tagged_event_wait(event: torch.npu.ExternalEvent):
    tag = _get_tag_by_event(event)
    if tag is None:
        raise AssertionError(
            "call npu_tagged_event_wait failed while event tag is None, please use "
            "torchair.ops.npu_create_tagged_external_event(tag: str) to create event then do event wait!")
    return torch.ops.air.external_event_wait(tag)


def _npu_tagged_event_reset(event: torch.npu.ExternalEvent):
    tag = _get_tag_by_event(event)
    if tag is None:
        raise AssertionError(
            "call npu_tagged_event_reset failed while event tag is None, please use "
            "torchair.ops.npu_create_tagged_external_event(tag: str) to create event then do event reset!")
    return torch.ops.air.external_event_reset(tag)


@torch.library.impl(lib, "external_event_record", "Meta")
def record_meta(tag: str):
    return None


@torch.library.impl(lib, "external_event_wait", "Meta")
def wait_meta(tag: str):
    return None


@torch.library.impl(lib, "external_event_reset", "Meta")
def reset_meta(tag: str):
    return None


def record_impl(tag: str):
    event = _GLOBAL_TAG_TO_EVENT.get(tag)
    if event is None:
        raise AssertionError(f"external event is None while tag is {tag}, "
                             f"external event record failed, please check the tag!")
    logger.debug("external event record with tag = [%s]", tag)
    return event.record()


def wait_impl(tag: str):
    event = _GLOBAL_TAG_TO_EVENT.get(tag)
    if event is None:
        raise AssertionError(f"external event is None while tag is {tag}, "
                             f"external event wait failed, please check the tag!")
    logger.debug("external event wait with tag = [%s]", tag)
    return event.wait()


def reset_impl(tag: str):
    event = _GLOBAL_TAG_TO_EVENT.get(tag)
    if event is None:
        raise AssertionError(f"external event is None while tag is {tag}, "
                             f"external event reset failed, please check the tag!")
    logger.debug("external event reset with tag = [%s]", tag)
    return event.reset()


torch.library.impl(lib, "external_event_record", "CompositeExplicitAutograd")(record_impl)
torch.library.impl(lib, "external_event_wait", "CompositeExplicitAutograd")(wait_impl)
torch.library.impl(lib, "external_event_reset", "CompositeExplicitAutograd")(reset_impl)


@register_fx_node_ge_converter(torch.ops.air.external_event_record.default)
def convert_event_record(tag: str,
                         meta_outputs: List[TensorSpec] = None):
    return None


@register_fx_node_ge_converter(torch.ops.air.external_event_wait.default)
def convert_event_wait(tag: str,
                       meta_outputs: List[TensorSpec] = None):
    return None


@register_fx_node_ge_converter(torch.ops.air.external_event_reset.default)
def convert_event_reset(tag: str,
                        meta_outputs: List[TensorSpec] = None):
    return None
