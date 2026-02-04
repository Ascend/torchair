import contextlib
import dataclasses
import logging
import sys
import types
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, Type

import torch

from torchair.core.utils import logger

logger.setLevel(logging.DEBUG)

"""Start to gen some API patch for AclGraph in st."""


# define stub FA API
def stub_npu_fa_func(*args, **kwargs):
    logger.debug('[Stub] using stub implementation of NPU FA with args: %s and kwargs: %s', args, kwargs)
    return torch.randn([3, 2])
    # return torch.empty_like(args[0])    示例实现


class StubNpuFA:
    def __init__(self):
        pass


stub_fa = StubNpuFA()
stub_fa.default = stub_npu_fa_func
stub_fa.out = stub_npu_fa_func


class StubConf:
    def __init__(self):
        self.allow_hf32 = 0
        pass


stub_conf = StubConf()

_GLOBAL_POOL_ID = 0


# define stub aclgraph API
def stub_graph_pool_handle():
    global _GLOBAL_POOL_ID
    _GLOBAL_POOL_ID += 1
    pool_id = (_GLOBAL_POOL_ID, 0)
    logger.debug('[Stub] run stub API graph_pool_handle, and return pool id is %s.', pool_id)
    return pool_id


def stub_synchronize():
    logger.debug('[Stub] run stub API stream synchronize with args[].')
    pass


def stub_empty_cache():
    logger.debug('[Stub] run stub API empty_cache')
    pass


def stub_get_npu_format(*args):
    logger.debug('[Stub] run stub API get_npu_format')
    return 2


def stub_get_deterministic_level():
    logger.debug('[Stub] run stub API get_deterministic_level')
    return 0


def stub_is_initialized(*args):
    logger.debug('[Stub] run stub API _is_initialized')
    pass


@contextlib.contextmanager
def stub_stream(stream=None):
    """Stub function for stream context manager."""
    if stream is not None:
        logger.debug(f"Stub: Pretending to switch to stream [%s]", stream)
    yield


class PatchAttr:
    def __init__(self, obj, attr_name, new_value):
        self.obj = obj
        self.attr_name = attr_name
        self.new_value = new_value
        self.original_value = None

    def __enter__(self):
        if hasattr(self.obj, self.attr_name):
            self.original_value = getattr(self.obj, self.attr_name)
            setattr(self.obj, self.attr_name, self.new_value)
        else:
            raise AttributeError(f"{self.obj} does not have attribute {self.attr_name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.obj, self.attr_name, self.original_value)


def raise_exception(*args, **kwargs):
    raise Exception("Should not be called")


@contextlib.contextmanager
def forbidden_attr(obj, attr_name):
    with PatchAttr(obj, attr_name, raise_exception):
        yield


@dataclasses.dataclass
class InputMeta:
    data: torch.Tensor
    is_prompt: bool


class StubNPUGraph:
    def __init__(self):
        logger.debug('[Stub] new stub class NPUGraph.')
        pass

    def capture_begin(self, pool=None, capture_error_mode="global"):
        logger.debug('[Stub] run stub API capture_begin with args[].')
        pass

    def capture_end(self):
        logger.debug('[Stub] run stub API capture_end with args[].')
        pass

    def replay(self):
        logger.debug('[Stub] run stub API replay with args[].')
        pass

    def reset(self):
        logger.debug('[Stub] run stub API reset with args[].')
        pass

    def pool(self):
        logger.debug('[Stub] run stub API pool with args[].')
        return (0, 1)


class graph:
    def __init__(
            self,
            npu_graph,
            pool=None,
            stream=None,
            capture_error_mode: str = "global"):
        logger.debug('[Stub] new stub class graph with args[%s, %s, %s, %s].',
                     type(npu_graph), pool, stream, capture_error_mode)
        self.stream_ctx = stub_stream(stream)
        self.npu_graph = npu_graph
        self.pool = () if pool is None else (pool,)
        self.stream = stream
        self.capture_error_mode = capture_error_mode
        pass

    def __enter__(self):
        torch.npu.synchronize()
        import gc
        gc.collect()
        torch.npu.empty_cache()
        self.stream_ctx.__enter__()

        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class StubStream:
    _counter = 0

    def __new__(cls, device=None, priority=0, **kwargs):
        stream_id = f"stream{cls._counter}"
        cls._counter += 1
        logger.debug('[Stub] new stub class Stream %s.', stream_id)
        return super().__new__(cls)

    def __init__(self, device=None, priority=0, **kwargs) -> None:
        self.stream_id = StubStream._counter

    def wait_event(self, event):
        logger.debug('[Stub] run stub API Stream::wait_event.')

    def wait_stream(self, stream):
        logger.debug('[Stub] run stub API Stream::wait_stream.')

    def record_event(self, event=None):
        logger.debug('[Stub] run stub API Stream::record_event.')

    def synchronize(self, event=None):
        logger.debug('[Stub] run stub API Stream::synchronize.')


class StubStreams:

    def Stream(self, event):
        return StubStream()


def current_stream(device=None):
    logger.debug('[Stub] run stub API current_stream.')
    return StubStream()


def get_stream_limit(stream):
    logger.debug('[Stub] run stub API get_stream_limit.')
    return {"cube_core_num": 24, 'vector_core_num': 48}


def set_stream_limit(stream, aicore_num, vectorcore_num):
    logger.debug('[Stub] run stub API set_stream_limit.')
    return "set_stream_limit"


def set_stream(stream):
    logger.debug('[Stub] run stub API set_stream.')
    return "set_stream"


def record_stream():
    logger.debug('[Stub] run stub API record_stream.')
    return "record_stream"


def current_device():
    logger.debug('[Stub] run stub API current_device.')
    return 0
    # 无恢复操作


def memory_snapshot():
    segments = [
        {'device': 0, 'address': 140366409891840, 'total_size': 1342177280, 'allocated_size': 0, 'requested_size': 0,
         'stream': 146751312, 'segment_type': 'large', 'segment_pool_id': (0, 1), 'is_expandable': False, 'frames': [],
         'blocks': [
             {'address': 140366409891840, 'size': 1342177280, 'requested_size': 1342177280, 'state': 'inactive',
              'frames': []}]
         },
        {'device': 0, 'address': 140370157502464, 'total_size': 2097152, 'allocated_size': 512,
         'requested_size': 8, 'stream': 0, 'segment_type': 'small', 'segment_pool_id': (0, 0), 'is_expandable': False,
         'frames': [], 'blocks': [
            {'address': 140370157502464, 'size': 512, 'requested_size': 8, 'state': 'active_allocated', 'frames': []},
            {'address': 140370157503488, 'size': 2096128, 'requested_size': 0, 'state': 'inactive', 'frames': []}]
         }
    ]
    return segments


class StubEvent:
    def __new__(cls):
        return super().__new__(cls)

    def record(self, stream=None):
        logger.debug('[Stub]Event record.')
        return

    def wait(self, stream=None):
        logger.debug('[Stub]Event wait.')
        return

    def reset(self, stream=None):
        logger.debug('[Stub]Event reset.')
        return


class _AutocastModule(types.ModuleType):
    autocast = lambda *_, **__: lambda f: f


class StubAmp:
    def __init__(self):
        logger.debug('[Stub] stub amp module.')

    def __new__(cls):
        return super().__new__(cls)

    def autocast(self):
        logger.debug('[Stub]Amp autocast')
        return


StubAmp.autocast_mode = _AutocastModule


class Stub_C:
    def __init__(self):
        logger.debug('[Stub] new stub _C Module.')
        pass

    @staticmethod
    def _npu_getCheckpointState(device, pool):
        logger.debug('[Stub] run stub API _npu_getCheckpointState with args[%s, %s].', device, pool)
        return "stub_mem_state"

    @staticmethod
    def _npu_setCheckpointPoolState(device, mem_state, stale_storages_ptr, storages_to_add_deleters_to_ptr):
        logger.debug('[Stub] run stub API _npu_setCheckpointPoolState to mem state %s.', mem_state)
        return

    @staticmethod
    def _construct_storage_from_data_pointer(data_ptr, device, nbytes):
        logger.debug('[Stub] run stub API _construct_storage_from_data_pointer with storage nbytes %s.', nbytes)
        return torch.Storage(nbytes)

    @staticmethod
    def _construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage):
        logger.debug('[Stub] run stub API _construct_NPU_Tensor_From_Storage_And_Metadata with metadata.')
        return torch.empty_strided(metadata['size'], metadata['stride'],
                                   dtype=metadata['dtype'],
                                   device=metadata['device'])


def _stub_npu_add_rms_norm_default(self, *args, **kwargs):
    return torch.randn([3, 2]), None, torch.randn([3, 2])


def _stub_npu_dynamic_quant_default(self, *args, **kwargs):
    return torch.randn([3, 2]), torch.randn([3, 2])


def _stub_npu_dtype_cast_default(self, *args, **kwargs):
    return torch.randn([3, 2]), torch.randn([3, 2])


# define stub submodule
class StubNpu:
    def __init__(self):
        logger.debug('[Stub] new stub module npu.')
        self.npu_fused_infer_attention_score = stub_fa
        self._npu_fused_infer_attention_score_get_max_workspace = stub_fa
        self.npu_fused_infer_attention_score_v2 = stub_fa
        self._npu_fused_infer_attention_score_v2_get_max_workspace = stub_fa
        self.NPUGraph = StubNPUGraph
        self.graph = graph
        self.Stream = StubStream
        self.streams = StubStreams
        self.Event = StubEvent
        self.current_stream = current_stream
        self.set_stream_limit = set_stream_limit
        self.get_stream_limit = get_stream_limit
        self.set_stream = set_stream
        self.current_device = current_device
        self.stream = stub_stream
        self.graph_pool_handle = stub_graph_pool_handle
        self.synchronize = stub_synchronize
        self.empty_cache = stub_empty_cache
        self.memory_snapshot = memory_snapshot
        self.matmul = stub_conf
        self.conv = stub_conf
        self._C = Stub_C
        self.amp = StubAmp
        self.is_initialized = stub_is_initialized
        self._get_deterministic_level = stub_get_deterministic_level
        self.npu_add_rms_norm = types.SimpleNamespace(
            default=_stub_npu_add_rms_norm_default
        )
        self.npu_dynamic_quant = types.SimpleNamespace(
            default=_stub_npu_dynamic_quant_default
        )
        self._npu_dtype_cast = types.SimpleNamespace(
            default=_stub_npu_dtype_cast_default
        )


def patch_ops_npu_module(stub_module):
    original_module = None
    original_exists = hasattr(torch.ops, 'npu')
    if original_exists:
        original_module = torch.ops.npu

    torch.ops.npu = stub_module
    logger.debug('[Stub] Original torch.ops.npu module is replaced by stub implementation: %s', torch.ops.npu)
    return original_module


def patch_torch_point_npu_module(stub_module):
    original_module = None
    original_exists = hasattr(torch, 'npu')
    if original_exists:
        original_module = torch.npu

    torch.npu = stub_module
    logger.debug('[Stub] Original torch.npu module is replaced by stub implementation: %s', torch.npu)
    return original_module


def patch_torch_npu_module(stub_module):
    original_module = None
    if 'torch_npu' in sys.modules:
        original_module = sys.modules['torch_npu']

    module = types.ModuleType('torch_npu_stub')
    module.npu = stub_module
    module._C = stub_module._C
    module.get_npu_format = stub_get_npu_format
    module.__all__ = ['npu']

    sys.modules['torch_npu'] = module
    logger.debug('[Stub] Original torch_npu.npu module is replaced by stub implementation: %s',
                 sys.modules['torch_npu'])
    return original_module


### register npu custom ops
def custom_infer_meta(data, other):
    torch._check(data.dim() == 4, lambda: "data rank must be 4")

    out_dim0 = (data.size(0) + 8) // 2
    out_dim1 = (data.size(1) * data.size(1)) // data.size(0)
    out_dim2 = data.size(2) - 1
    out_dim3 = data.size(3)
    tmp_out = torch.empty([out_dim0, out_dim1, out_dim2, out_dim3], dtype=data.dtype, device=data.device)
    return tmp_out


def custom_infer_npu(data, other):
    torch._check(data.dim() == 4, lambda: "data rank must be 4")

    out_dim0 = (data.size(0) + 8) // 2
    out_dim1 = (data.size(1) * data.size(1)) // data.size(0)
    out_dim2 = data.size(2) - 1
    out_dim3 = data.size(3)
    tmp_out = torch.randn([out_dim0, out_dim1, out_dim2, out_dim3], dtype=data.dtype, device=data.device)
    return tmp_out


def custom_infer_out_meta(data, other, *, out):
    return out


def custom_infer_out_npu(data, other, *, out):
    out.fill_(1)
    return out


### register npu custom inplace ops for testing reinplace fx pass with multiple inplace args
def sin_cos_inplace_meta(x, out_sin, out_cos):
    return torch.empty_like(x)


def sin_cos_inplace(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> torch.Tensor:
    out_sin.sin_()
    out_cos.cos_()
    return x + 1


def sin_cos_functional_meta(x, out_sin, out_cos):
    return torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)


def sin_cos_functional(x: torch.Tensor,
                       sin: torch.Tensor,
                       cos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sin_clone = sin.clone()
    cos_clone = cos.clone()
    res = torch.ops.custom.sin_cos_inplace(x, sin_clone, cos_clone)
    return res, sin_clone, cos_clone


lib = torch.library.Library("custom", "FRAGMENT")

_custom_ops_registered = False


def register_custom_ops():
    global _custom_ops_registered
    if _custom_ops_registered:
        return

    _custom_ops_registered = True

    if not hasattr(torch.ops.custom, "custom_infer"):
        lib.define("custom_infer(Tensor data, Tensor other) -> Tensor")
        torch.library.impl(lib, "custom_infer", "Meta")(custom_infer_meta)
        torch.library.impl(lib, "custom_infer", "CompositeExplicitAutograd")(custom_infer_npu)

    if not hasattr(torch.ops.custom, "custom_infer.out"):
        lib.define("custom_infer.out(Tensor data, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")
        torch.library.impl(lib, "custom_infer.out", "Meta")(custom_infer_out_meta)
        torch.library.impl(lib, "custom_infer.out", "CompositeExplicitAutograd")(custom_infer_out_npu)

    if not hasattr(torch.ops.custom, "sin_cos_inplace"):
        lib.define("sin_cos_inplace(Tensor x, Tensor(a!) out_sin, Tensor(b!) out_cos) -> Tensor")
        torch.library.impl(lib, "sin_cos_inplace", "Meta")(sin_cos_inplace_meta)
        torch.library.impl(lib, "sin_cos_inplace", "CompositeExplicitAutograd")(sin_cos_inplace)

    if not hasattr(torch.ops.custom, "sin_cos_functional"):
        lib.define("sin_cos_functional(Tensor x, Tensor out_sin, Tensor out_cos) -> (Tensor, Tensor, Tensor)")
        torch.library.impl(lib, "sin_cos_functional", "Meta")(sin_cos_functional_meta)
        torch.library.impl(lib, "sin_cos_functional", "CompositeExplicitAutograd")(sin_cos_functional)

        @torch.library.impl(lib, "sin_cos_inplace", "Functionalize")
        def sin_cos_inplace_functionalize(x, sin, cos):
            torch._sync(x)
            torch._sync(sin)
            torch._sync(cos)
            x_wrap = torch._from_functional_tensor(x)
            sin_wrap = torch._from_functional_tensor(sin)
            cos_wrap = torch._from_functional_tensor(cos)
            with torch._C._ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            ):
                res, sin_out, cos_out = torch.ops.custom.sin_cos_functional(x_wrap, sin_wrap, cos_wrap)
            torch._functionalize_replace(sin, sin_out)
            torch._functionalize_replace(cos, cos_out)
            torch._functionalize_commit_update(sin)
            torch._functionalize_commit_update(cos)
            torch._sync(sin)
            torch._sync(cos)
            return res

    from torchair._acl_concrete_graph.graph_pass import (
        inplaceable_npu_ops,
        InplaceableNpuOp,
        check_multi_stream_for_multi_reinplace
    )
    inplaceable_npu_ops.update({
        torch.ops.custom.sin_cos_functional.default:
            InplaceableNpuOp(
                inplace_op=torch.ops.custom.sin_cos_inplace.default,
                mutated_arg=[1, 2],
                extra_check=check_multi_stream_for_multi_reinplace,
            ),
    })
