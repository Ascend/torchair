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


if not hasattr(getattr(torch.ops, "air"), "tagged_event_record"):
    lib.define("tagged_event_record(str tag, bool created_inside=False) -> ()")
    has_side_effect(torch.ops.air.tagged_event_record.default)

    @torch.library.impl(lib, "tagged_event_record", "Meta")
    def record_meta(tag: str, created_inside: bool = False):
        return None

    torch.library.impl(lib, "tagged_event_record", "CompositeExplicitAutograd")(record_impl)

    @register_fx_node_ge_converter(torch.ops.air.tagged_event_record.default)
    def convert_event_record(tag: str,
                             created_inside: bool = False,
                             meta_outputs: List[TensorSpec] = None):
        raise NotImplementedError("torch.ops.air.tagged_event_record.default ge_converter is not implemented!")

if not hasattr(getattr(torch.ops, "air"), "tagged_event_wait"):
    lib.define("tagged_event_wait(str tag, bool created_inside=False) -> ()")
    has_side_effect(torch.ops.air.tagged_event_wait.default)

    @torch.library.impl(lib, "tagged_event_wait", "Meta")
    def wait_meta(tag: str, created_inside: bool = False):
        return None

    torch.library.impl(lib, "tagged_event_wait", "CompositeExplicitAutograd")(wait_impl)

    @register_fx_node_ge_converter(torch.ops.air.tagged_event_wait.default)
    def convert_event_wait(tag: str,
                           created_inside: bool = False,
                           meta_outputs: List[TensorSpec] = None):
        raise NotImplementedError("torch.ops.air.tagged_event_wait.default ge_converter is not implemented!")

if not hasattr(getattr(torch.ops, "air"), "record_tagged_stream"):
    lib.define("record_tagged_stream(Tensor input, str tag) -> ()")
    lib.define("record_tagged_stream_(Tensor(a!) input, str tag) -> ()")

    has_side_effect(torch.ops.air.record_tagged_stream.default)

    @torch.library.impl(lib, "record_tagged_stream", "Meta")
    def record_tagged_stream_meta(input: torch.Tensor, tagged_stream: str):
        return None

    @torch.library.impl(lib, "record_tagged_stream_", "Meta")
    def record_tagged_stream_inplace_meta(input: torch.Tensor, tagged_stream: str):
        return None

    @torch.library.impl(lib, "record_tagged_stream_", "Functionalize")
    def record_tagged_stream_inplace_func(input: torch.Tensor, tagged_stream: str):
        # The record_stream interface does not involve input mutation,
        # so there is no need to copy the output of out-of-place op to the original input.
        torch.ops.air.record_tagged_stream(input, tagged_stream)

    torch.library.impl(lib, "record_tagged_stream", "CompositeExplicitAutograd")(record_tagged_stream_impl)
    torch.library.impl(lib, "record_tagged_stream_", "CompositeExplicitAutograd")(record_tagged_stream_impl)

    @register_fx_node_ge_converter(torch.ops.air.record_tagged_stream.default)
    def convert_record_tagged_stream(input: torch.Tensor,
                                     tagged_stream: str,
                                     meta_outputs: List[TensorSpec] = None):
        raise NotImplementedError("torch.ops.air.record_tagged_stream.default ge_converter is not implemented!")

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


# ============================================================================
# New operators for stream/event operations with specified stream
# ============================================================================

def _resolve_stream_markers(stream_id, device_index, device_type):
    """
    将 stream 参数转换为 int。如果参数是 int 或 torch.fx.Node，直接转为 int。

    Default stream 标记字符串的处理由 resolve_default_stream_markers pass 完成。

    Args:
        stream_id: int 或其他可转换为 int 的值
        device_index: int 或其他转换为 int 的值
        device_type: int 或其他可转换为 int 的值

    Returns:
        tuple: (stream_id: int, device_index: int, device_type: int)
    """
    return int(stream_id), int(device_index), int(device_type)


def record_on_stream_impl(event_tag: str, stream_id, device_index, device_type, created_inside: bool = False):
    # 解析 stream 标记（如果是标记字符串则动态获取当前流信息）
    stream_id, device_index, device_type = _resolve_stream_markers(stream_id, device_index, device_type)

    event = _get_event_by_tag(event_tag, created_inside)
    if event is None:
        custom_warning = 'please report an issue to Torchair!'
        inside_warning = f'please make sure you have created tag {event_tag} with npu_create_tagged_event API.'
        raise AssertionError(f"tagged event is None while tag is {event_tag}, "
                             f"tagged event record on stream failed, "
                             f"{inside_warning if created_inside else custom_warning}")
    stream = torch.npu.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)
    logger.debug("tagged event record on stream with tag = [%s], stream_id = [%d]", event_tag, stream_id)
    return event.record(stream)


def wait_on_stream_impl(event_tag: str, stream_id, device_index, device_type, created_inside: bool = False):
    # 解析 stream 标记（如果是标记字符串则动态获取当前流信息）
    stream_id, device_index, device_type = _resolve_stream_markers(stream_id, device_index, device_type)

    event = _get_event_by_tag(event_tag, created_inside)
    if event is None:
        custom_warning = 'please report an issue to Torchair!'
        inside_warning = f'please make sure you have created tag {event_tag} with npu_create_tagged_event API.'
        raise AssertionError(f"tagged event is None while tag is {event_tag}, "
                             f"tagged event wait on stream failed, "
                             f"{inside_warning if created_inside else custom_warning}")
    stream = torch.npu.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)
    logger.debug("tagged event wait on stream with tag = [%s], stream_id = [%d]", event_tag, stream_id)
    return event.wait(stream)


def stream_wait_stream_impl(stream_id, device_index, device_type,
                            other_stream_id, other_device_index, other_device_type):
    # 解析 stream 标记（如果是标记字符串则动态获取当前流信息）
    stream_id, device_index, device_type = _resolve_stream_markers(stream_id, device_index, device_type)
    other_stream_id, other_device_index, other_device_type = _resolve_stream_markers(
        other_stream_id, other_device_index, other_device_type)

    stream = torch.npu.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)
    other_stream = torch.npu.Stream(stream_id=other_stream_id, device_index=other_device_index, device_type=other_device_type)
    logger.debug("stream wait stream with stream_id = [%d], other_stream_id = [%d]", stream_id, other_stream_id)
    return stream.wait_stream(other_stream)


# Register tagged_event_record_on_stream operator
if not hasattr(getattr(torch.ops, "air"), "tagged_event_record_on_stream"):
    lib.define("tagged_event_record_on_stream(str event_tag, str stream_id, str device_index, str device_type, bool created_inside=False) -> ()")
    has_side_effect(torch.ops.air.tagged_event_record_on_stream.default)

    @torch.library.impl(lib, "tagged_event_record_on_stream", "Meta")
    def record_on_stream_meta(event_tag: str, stream_id, device_index, device_type, created_inside: bool = False):
        return None

    torch.library.impl(lib, "tagged_event_record_on_stream", "CompositeExplicitAutograd")(record_on_stream_impl)

# Register tagged_event_wait_on_stream operator
if not hasattr(getattr(torch.ops, "air"), "tagged_event_wait_on_stream"):
    lib.define("tagged_event_wait_on_stream(str event_tag, str stream_id, str device_index, str device_type, bool created_inside=False) -> ()")
    has_side_effect(torch.ops.air.tagged_event_wait_on_stream.default)

    @torch.library.impl(lib, "tagged_event_wait_on_stream", "Meta")
    def wait_on_stream_meta(event_tag: str, stream_id, device_index, device_type, created_inside: bool = False):
        return None

    torch.library.impl(lib, "tagged_event_wait_on_stream", "CompositeExplicitAutograd")(wait_on_stream_impl)

# Register tagged_stream_wait_stream operator
if not hasattr(getattr(torch.ops, "air"), "tagged_stream_wait_stream"):
    lib.define("tagged_stream_wait_stream(str stream_id, str device_index, str device_type, str other_stream_id, str other_device_index, str other_device_type) -> ()")
    has_side_effect(torch.ops.air.tagged_stream_wait_stream.default)

    @torch.library.impl(lib, "tagged_stream_wait_stream", "Meta")
    def stream_wait_stream_meta(stream_id, device_index, device_type,
                                other_stream_id, other_device_index, other_device_type):
        return None

    torch.library.impl(lib, "tagged_stream_wait_stream", "CompositeExplicitAutograd")(stream_wait_stream_impl)
