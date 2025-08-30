import contextlib
import dataclasses
import logging
import os
import sys
import types
from typing import List
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
from torchair._acl_concrete_graph.utils import reconstruct_args_kwargs, WeakRef
from torchair_st_utils import capture_stdout, capture_logger

logger.setLevel(logging.DEBUG)

"""Start to gen some API patch for AclGraph in st."""


# define stub FA API
def stub_npu_fa_func(*args, **kwargs):
    logger.debug('[Stub] using stub implementation of NPU FA with args: %s and kwargs: %s', args, kwargs)
    return torch.randn([3, 2])
    return torch.empty_like(args[0])  # 示例实现


class StubNpuFA:
    def __init__(self):
        pass


stub_fa = StubNpuFA()
stub_fa.default = stub_npu_fa_func
stub_fa.out = stub_npu_fa_func

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
        return stream_id

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        pass


def current_stream(device=None):
    logger.debug('[Stub] run stub API current_stream.')
    return "current_stream"


def set_stream(stream):
    logger.debug('[Stub] run stub API set_stream.')
    return "set_stream"


def record_stream():
    logger.debug('[Stub] run stub API record_stream.')
    return "record_stream"


def current_device():
    logger.debug('[Stub] run stub API current_device.')
    return "current_device"
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
        self.Event = StubEvent
        self.current_stream = current_stream
        self.set_stream = set_stream
        self.current_device = current_device
        self.stream = stub_stream
        self.graph_pool_handle = stub_graph_pool_handle
        self.synchronize = stub_synchronize
        self.empty_cache = stub_empty_cache
        self.memory_snapshot = memory_snapshot
        self._C = Stub_C


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
    module.__all__ = ['npu']

    sys.modules['torch_npu'] = module
    logger.debug('[Stub] Original torch_npu.npu module is replaced by stub implementation: %s',
                 sys.modules['torch_npu'])
    return original_module


_get_pool_id = None


class AclGraphSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak
        return super().tearDown()

    def test_aclgraph_capture_and_replay(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)
        x = torch.randn([3, 2])
        for i in range(2):
            model(x)

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.mul_(2)
                return x + 1

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # inference
        with self.assertLogs(logger, level="WARNING") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("data_ptr is different between capture and replay." in log for log in cm.output),
            f"Expected WARNING 'Mutated input[arg]'s data_ptr is different between capture and replay.' "
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_false(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.div_(2)
                return x - 1

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # expected no warning called
        from unittest.mock import patch
        with patch("logging.Logger.warning") as mock_warning:
            for _ in range(2):
                output = model(x)
            mock_warning.assert_not_called()

    def test_aclgraph_update(self):
        from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
        _REPLACE_FUNC_MAP[torch.ops.aten.max_unpool2d.default] = StaticWorkspaceReplaceFunc(
            get_workspace=None,
            out_operator=torch.ops.aten.max_unpool2d.out,
            workspace_keys=[],
            output_keys=["out"],
            updated_param_keys=[],
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, output_size):
                val = torch.nn.functional.max_unpool2d(x, y, output_size)
                return val.mean()

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        output, indices = F.max_pool1d(
            torch.randn([1, 1, 4]), 2, stride=2, return_indices=True
        )

        torch._dynamo.mark_static(output)
        torch._dynamo.mark_static(indices)
        model(output, indices, 2)
        model(output, indices, 2)

    def test_aclgraph_dynamic_sym_in_tensor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, input):
                ln1 = self.linear1(input)
                ln2 = self.linear2(input)
                return ln1 + ln2

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([4, 3])
        res = model(x)
        first_model_id = id(model)
        x2 = torch.randn([5, 3])
        res = model(x2)
        second_model_id = id(model)
        self.assertTrue(first_model_id == second_model_id)

    def test_aclgraph_dynamic_sym_in_scale_and_tensor(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        def get_graph_num(concrete_graph):
            return len(concrete_graph.graph.graph)

        def wrapper_call(func, start_func_num, add_graph_num):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph_num = get_graph_num(args[0])
                assert graph_num == start_func_num, \
                    f"before call, assert graph num failed, expect {start_func_num}, get {graph_num}"

                ret = func(*args, **kwargs)

                graph_num = get_graph_num(args[0])
                assert graph_num == start_func_num + add_graph_num, \
                    f"after call, assert graph num failed, expect {start_func_num + add_graph_num}, get {graph_num}"
                return ret

            return wrapper

        bak_func = AclConcreteGraph.__call__
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 0, 1)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x, s):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1, torch.add(ln2, s)

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.debug.aclgraph.enable_output_clone = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        x = torch.randn([5, 2])
        scale1 = 4
        torch._dynamo.reset()
        model(x, scale1)

        # no find captured graph, capture another npu graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 1, 1)
        with capture_logger() as stdout:
            scale1 = 5
            model(x, scale1)
        captured_output = stdout.getvalue()
        self.assertTrue("After setting to original memory state for fx_graph" in captured_output)
        self.assertTrue("No find captured AclGraph" in captured_output)

        # find captured graph, no need to capture another npu graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 2, 0)
        with capture_logger() as stdout:
            scale1 = 4
            model(x, scale1)
        captured_output = stdout.getvalue()
        self.assertTrue("Find captured AclGraph" in captured_output)

        # original fx graph, but no this graph key, need to capture graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 2, 1)
        with capture_logger() as stdout:
            scale1 = 4
            x2 = torch.randn([6, 2])
            model(x2, scale1)
        captured_output = stdout.getvalue()
        self.assertTrue("No find captured AclGraph" in captured_output)
        AclConcreteGraph.__call__ = bak_func

        # another fx graph, need to capture graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 0, 1)
        torch._dynamo.reset()
        with capture_logger() as stdout:
            scale1 = 5.0
            model(x2, scale1)
        captured_output = stdout.getvalue()
        self.assertTrue("No find captured AclGraph" in captured_output)
        AclConcreteGraph.__call__ = bak_func

    def test_aclgraph_unsupported_dump(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x - 1.0

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.graph_dump.type = "pbtxt"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)
        x = torch.randn([3, 2])
        with self.assertRaisesRegex(
                RuntimeError,
                r"Graph dump for aclGraph only support 'py' type, but got: pbtxt"
        ):
            model(x)

    def test_eager_with_multi_stream_event(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_event(tag="33")

            def forward(self, x):
                def branch1(xx):
                    y = xx + 1
                    y = y * y
                    y = y @ y
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return y

                def branch2(xx):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    y = xx - 1
                    y = y @ y
                    return y

                with torch.npu.stream(self.stream1):
                    out1 = branch1(x)
                with torch.npu.stream(self.stream2):
                    out2 = branch2(x)
                return out1, out2

        model = Model()
        x = torch.randn([3, 3])
        for _ in range(2):
            model(x)

    def test_ge_with_multi_stream_event(self):
        from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        # ignore event in ge mode
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ext_event = torchair.ops.npu_create_tagged_event(tag="44")

            def forward(self, xx):
                y = xx + xx
                y = y @ y
                y = y * y
                y = y + y
                torchair.ops.npu_tagged_event_record(self.ext_event)
                torchair.ops.npu_tagged_event_wait(self.ext_event)
                return y

        def check_graph(concrete_graph):
            # fx graph has npu_tagged_event_record\npu_tagged_event_wait\npu_tagged_event_reset while ge graph does not
            # ge graph has netoutput, but fx_graph does not have
            # so len(concrete_graph.graph.op) == 7, len(concrete_graph.fx_graph.graph.nodes) == 9
            assert len(concrete_graph.graph.op) == 7, f"expect op count is 9, but got {len(concrete_graph.graph.op)}"
            assert len(concrete_graph.fx_graph.graph.nodes) == 9, \
                f"expect node count is 9, but got {len(concrete_graph.fx_graph.graph.nodes)}"
            return

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) > 1
                check_graph(args[0])
                return tuple([args[1]])

            return wrapper

        GeConcreteGraph.__call__ = decorator(GeConcreteGraph.__call__)

        model = Model()
        opt_model = torch.compile(model, backend=npu_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        for _ in range(2):
            try:
                opt_model(x)
            except Exception as e:
                assert str(e).__contains__("torch.ops.air.tagged_event_record.default ge_converter is not implemented")

    def test_npu_stream_switch_with_stream_closure(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.npu_stream_switch('1', 3):
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.npu_stream_switch('2', 3):
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            opt_model(x, x, x, x)

        self.assertTrue(
            any("Try to capture node names[tagged_event_record_default] type[air.tagged_event_record.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_default] "
            f"type[air.tagged_event_record.default]' in logs: {cm.output}")
        # stream tag 1
        self.assertTrue(
            any("guard with user stream scope, node = tagged_event_wait_default, user stream label = 1"
                in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = tagged_event_wait_default, "
            f"user stream label = 1' in logs: "
            f"{cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_wait_default] type[air.tagged_event_wait.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        # stream tag 2
        self.assertTrue(
            any("guard with user stream scope, node = tagged_event_wait_default_1, user stream label = 2"
                in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = tagged_event_wait_default_1, "
            f"user stream label = 2' in logs:"
            f" {cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_wait_default_1] type[air.tagged_event_wait.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_1] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        # mm在stream tag 1上执行
        self.assertTrue(
            any("guard with user stream scope, node = mm, user stream label = 1" in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = mm, user stream label = 1' in logs:"
            f" {cm.output}")

        # mm_1 在stream tag 2上执行
        self.assertTrue(
            any("guard with user stream scope, node = mm_1, user stream label = 2" in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = mm_1, user stream label = 2' in logs:"
            f" {cm.output}")

        # 两条从流stream分别向主capture流发送record-wait对，以完成event闭环
        # stream tag 2
        self.assertTrue(
            any("guard with user stream scope, node = tagged_event_record_default_1, user stream label = 2"
                in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = tagged_event_record_default_1, "
            f"user stream label = 2' in logs:"
            f" {cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_record_default_1] type[air.tagged_event_record.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_default_1] "
            f"type[air.tagged_event_record.default]' in logs: {cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_wait_default_2] type[air.tagged_event_wait.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_2] "
            "type[air.tagged_event_wait.default]' in logs: {cm.output}")

        # stream tag 1
        self.assertTrue(
            any("guard with user stream scope, node = tagged_event_record_default_2, user stream label = 1"
                in log for log in cm.output),
            f"Expected no DEBUG log 'guard with user stream scope, node = tagged_event_record_default_2, "
            f"user stream label = 1' "
            f"in logs: {cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_record_default_2] type[air.tagged_event_record.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_default_2] "
            f"type[air.tagged_event_record.default]' in logs: {cm.output}")
        self.assertTrue(
            any("Try to capture node names[tagged_event_wait_default_3] type[air.tagged_event_wait.default]"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_3] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")


    def test_npu_stream_switch_with_tagged_event(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        ext_event1 = torchair.ops.npu_create_tagged_event(tag="66")
        ext_event2 = torchair.ops.npu_create_tagged_event(tag="77")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                torchair.ops.npu_tagged_event_record(ext_event1)
                torchair.ops.npu_tagged_event_record(ext_event2)
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.npu_tagged_event_wait(ext_event1)
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.npu_stream_switch('2', 3):
                        torchair.ops.npu_tagged_event_wait(ext_event2)
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        def check_graph(concrete_graph):
            event_record = 0
            for node in concrete_graph.fx_graph.graph.nodes:
                if str(node.target) == "aten.mm.default":
                    assert str(node.prev.target) == "air.tagged_event_wait.default"
                if str(node.target) == "air.tagged_event_record.default":
                    event_record += 1
            assert event_record == 5, f"expect event record count is 5, but got {event_record}"

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) >= 3
                check_graph(args[0])
                return tuple([args[0], args[1], args[2]])

            return wrapper

        AclConcreteGraph.__call__ = decorator(AclConcreteGraph.__call__)

        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        opt_model(x, y, z, w)

    def test_npu_stream_switch_no_support_npu_wait_tensor_with_reduce_over_head(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.scope.npu_wait_tensor(in4, add_result)
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.npu_stream_switch('2', 3):
                        torchair.scope.npu_wait_tensor(in3, add_result)
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        model = Model()
        config_view = CompilerConfig()
        config_view.mode = "reduce-overhead"
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        in3 = torch.randn(2, 2)
        in4 = torch.randn(2, 2)
        try:
            model(in1, in2, in3, in4)
        except Exception as e:
            assert str(e).__contains__("torch.ops.air.wait_tensor kernel_impl is not implemented! "
                                       "if you are using torch.compile")

    def test_record_stream_with_reduce_over_head(self):

        class StubTensor:
            def record_stream(self, stream):
                return

        origin = torch.Tensor.record_stream
        torch.Tensor.record_stream = StubTensor.record_stream

        def func():
            A = torch.ones([100, 100])
            mm_input = torch.randn(3200, 32000)
            with torchair.scope.npu_stream_switch('1', 3):
                for _ in range(10):  # 延长secend stream执行时间，使得A.add(1)晚于主流C.add_(2)计算
                    out = mm_input * mm_input
                B = A.add(1)
                torchair.ops.npu_record_tagged_stream(B, '1')
            del A
            C = torch.ones([100, 100])
            C.add_(2)
            return B, C

        config_view = CompilerConfig()
        config_view.mode = "reduce-overhead"
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(func, backend=npu_backend_view, dynamic=False)

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model()

        self.assertTrue(
            any("call_function[target=torch.ops.air.record_tagged_stream.default]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.air.record_tagged_stream.default]' in logs: {cm.output}"
        )
        torch.Tensor.record_stream = origin

    def test_reinplace_pass_disblabled_with_multi_stream(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = a.add(1)
                with torchair.scope.npu_stream_switch('1', 3):
                    y.mul_(2)
                return b, y

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertFalse(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

        self.assertFalse(
            any("call_function[target=torch.ops.aten.mul_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.mul_.Tensor]' in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true_default_enable_reinplace_pass(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = a.add(1)
                y.mul_(2)
                return b, y

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertTrue(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

        self.assertTrue(
            any("call_function[target=torch.ops.aten.mul_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.mul_.Tensor]' in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true_disable_reinplace_pass_with_slice(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x[:2]
                x.add_(5)
                x.mul_(7)
                return x

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x)

        self.assertTrue(
            any("processing reinplace_input_mutated_ops_pass" in log for log in cm.output),
            f"Expected DEBUG log 'processing reinplace_input_mutated_ops_pass' in logs: {cm.output}"
        )

    def test_aclgraph_keep_inference_input_mutations_true_disable_mutated_input_pass_with_slice(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x[:2]
                x.add_(1)
                x.mul_(3)
                return x

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x)

        self.assertTrue(
            any("processing reinplace_inplaceable_ops_pass" in log for log in cm.output),
            f"Expected DEBUG log 'processing reinplace_inplaceable_ops_pass' in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true_disable_reinplace_mutated_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = a.add(1)
                y.mul(3)
                return b, y

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertTrue(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' "
            f"not found in logs: {cm.output}"
        )

        self.assertFalse(
            any("call_function[target=torch.ops.aten.copy_.default]" in log for log in cm.output),
            f"Expected no DEBUG log 'call_function[target=torch.ops.aten.copy_.default]' "
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true_disable_reinplace_inplaceable_ops(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.mul_(2)
                x.add_(1)
                return x

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([5, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x)

        self.assertTrue(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' "
            f"not found in logs: {cm.output}"
        )

        self.assertTrue(
            any("call_function[target=torch.ops.aten.mul.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.mul.Tensor]' "
            f"not found in logs: {cm.output}"
        )

        # cannot erase copy_ node in this case, need ".out" fx pass, should be optimize in the future.
        self.assertTrue(
            any("call_function[target=torch.ops.aten.copy_.default]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.copy_.default]' "
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_false_enable_reinplace_ops(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = a.add_(1)
                y.mul(5)
                return b, x

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertTrue(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_false_disable_reinplace_ops(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = x.clone()
                b = x.add_(1)
                y.mul(6)
                return b, x

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=backend, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertFalse(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected no DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

    def test_graph_dump_with_py(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        def get_dumped_py_file_list(dir_path, file_extension='.py'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo_o') and i.endswith(f'{file_extension}')]

        for file_name in get_dumped_py_file_list('./'):
            os.remove(os.path.join('./', file_name))

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.graph_dump.type = "py"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        model = Model()
        model = torch.compile(model, backend=npu_backend)
        x = torch.randn(2, 2)
        model(x)

        dumped_py_file_list = get_dumped_py_file_list('./')
        dumped_py_file_list.sort(
            key=lambda file_name: os.path.getmtime(os.path.join('./', file_name)))
        assert dumped_py_file_list.__len__() > 0
        file_name = os.path.join('./', dumped_py_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()

        self.assertIn("torch.ops.aten.add.Tensor(arg0_1, 1)", src)
        self.assertIn("code: return x + 1", src)

        for file_name in get_dumped_py_file_list('./'):
            os.remove(os.path.join('./', file_name))

    def test_aclgraph_dynamic_output_construct_in_share_memory(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)

            def forward(self, input):
                ln1 = self.linear1(input)
                return ln1, ln1.view(-1), ln1[2:], 2

        torch._dynamo.reset()
        x = torch.randn([4, 2])
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model1 = Model()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res1, res2, res3, res4 = model1(x)
        self.assertTrue(res1.untyped_storage().data_ptr() == res2.untyped_storage().data_ptr())
        self.assertTrue(res1.untyped_storage().data_ptr() == res3.untyped_storage().data_ptr())
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)

        # same graph with valid output ref, no need reconstruct
        del res2, res3, res4
        with capture_logger() as stdout:
            res1, res2, res3, res4 = model1(x)
        self.assertTrue(res1.untyped_storage().data_ptr() == res2.untyped_storage().data_ptr())
        self.assertTrue(res1.untyped_storage().data_ptr() == res3.untyped_storage().data_ptr())
        captured_output = stdout.getvalue()
        self.assertFalse("no need to reconstruct output tensors for" in captured_output)  # should be true in real env

        # same graph with invalid output ref, need reconstruct
        del res1, res2, res3, res4
        with capture_logger() as stdout:
            res1, res2, res3, res4 = model1(x)
        self.assertTrue(res1.untyped_storage().data_ptr() == res2.untyped_storage().data_ptr())
        self.assertTrue(res1.untyped_storage().data_ptr() == res3.untyped_storage().data_ptr())
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)

    def test_aclgraph_dynamic_disable_mempool_reuse_in_same_fx(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, input, bias):
                ln1 = self.linear1(input)
                ln2 = self.linear2(input)
                return ln1, torch.add(ln2, bias)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = Model()
        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])

        torch._dynamo.reset()
        with capture_logger() as stdout:
            model(x, 9.9)
        captured_output = stdout.getvalue()
        self.assertTrue("memory pool reuse is disable" in captured_output)
        self.assertTrue("no mempool reuse in fx_graph" in captured_output)

    def test_aclgraph_dynamic_use_custom_pool(self):
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)

            def forward(self, input):
                ln1 = self.linear1(input)
                return ln1

        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, input):
                ln2 = self.linear2(input)
                return ln2 + 1

        x = torch.randn([3, 2])
        torch._dynamo.reset()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)

                assert len(args) > 0
                concrete_graph = args[0]
                global _get_pool_id
                _get_pool_id = concrete_graph.graph.pool
                return ret

            return wrapper

        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        # test no set custom pool, check different pool, and reconstruct outputs
        model1 = Model1()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)

        # same graph with valid output ref, no need reconstruct
        with capture_logger() as stdout:
            res = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("no need to reconstruct output tensors for" in captured_output)
        pool_id1 = _get_pool_id

        model2 = Model2()
        model2 = torch.compile(model2, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res = model2(x)
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)
        pool_id2 = _get_pool_id

        self.assertTrue(pool_id1 != pool_id2)

        # test set custom pool, check same pool, and no reconstruct outputs
        config.aclgraph_config.use_custom_pool = torch.npu.graph_pool_handle()
        aclgraph_backend2 = torchair.get_npu_backend(compiler_config=config)

        model1 = Model1()
        model1 = torch.compile(model1, backend=aclgraph_backend2, dynamic=True)
        with capture_logger() as stdout:
            res = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("no mempool reuse in fx_graph" in captured_output)
        pool_id1 = _get_pool_id

        model2 = Model2()
        model2 = torch.compile(model2, backend=aclgraph_backend2, dynamic=True)
        with capture_logger() as stdout:
            res = model2(x)
        captured_output = stdout.getvalue()
        self.assertTrue("no mempool reuse in fx_graph" in captured_output)
        pool_id2 = _get_pool_id

        self.assertTrue(pool_id1 == pool_id2)

    def test_reconstruct_args_kwargs(self):
        def _check_same_tensor_meta(x, y):
            if (list(x.shape) != list(y.shape)) or (
                    x.stride() != y.stride()) or (
                    x.device != y.device):
                return False
            else:
                return True

        def _check_same_list(list_x, list_y):
            if len(list_x) != len(list_y):
                return False

            res = []
            for idx, x_i in enumerate(list_x):
                if isinstance(x_i, torch.Tensor):
                    res.append(_check_same_tensor_meta(x_i, list_y[idx]))
                else:
                    res.append(x_i == list_y[idx])
            return all(res)

        args = [
            torch.randn([2, 3, 4, 5], dtype=torch.float16),
            torch.ones([2, 3, 4]).transpose(0, 1),
            torch.zeros([3, 24])[1:],
        ]

        kwargs = {
            "tag1": torch.randn(3, 4, 5),
            "tag2": [torch.empty([2, 3]), torch.empty([3, 4])],
            "tag3": "tag3_value",
            "tag40": 4,
            "tag41": 4.1,
            "tag42": False,
            "tag5": [2, 3, 4],
            "tag6": (True, 6.0),
            "tag7": (torch.empty([2, 3]), torch.empty([3, 4])),
            "tag8": [[2, 3], torch.empty([3, 4])],
        }

        out_args, out_kwargs = reconstruct_args_kwargs(args, kwargs)
        self.assertTrue(_check_same_list(args, out_args))
        for key in {"tag2", "tag5", "tag6", "tag7", "tag8"}:
            self.assertTrue(_check_same_list(kwargs[key], out_kwargs[key]))
        for key in {"tag1"}:
            self.assertTrue(_check_same_tensor_meta(kwargs[key], out_kwargs[key]))
        for key in {"tag3", "tag40", "tag41", "tag42"}:
            self.assertTrue(kwargs[key] == out_kwargs[key])

    def test_weak_ref(self):
        a = torch.randn(2, 3)
        b = torch.randn(4, 5)
        c = 1.0
        d = ["x", "y", "z"]

        ori_list = [a, b, c, d]
        weak_ref_list = [WeakRef(itr) for itr in ori_list]

        # check weak ref when all objs are alive
        ref_out = [ref() for ref in weak_ref_list]
        for idx, ref_i in enumerate(ref_out):
            if isinstance(ref_i, torch.Tensor):
                cosine_sim_val = F.cosine_similarity(ref_out[idx], ori_list[idx])
                self.assertTrue(cosine_sim_val.min().item() >= 0.9999)
            else:
                self.assertTrue(ref_out[idx] == ori_list[idx])
        del ref_out

        a2 = torch.randn(3, 2)
        ori_list[0] = a2
        weak_ref_list[0].swap_weakref(a2)

        # check weak ref when some weak obj swap
        ref_out = [ref() for ref in weak_ref_list]
        for idx, ref_i in enumerate(ref_out):
            if isinstance(ref_i, torch.Tensor):
                cosine_sim_val = F.cosine_similarity(ref_out[idx], ori_list[idx])
                self.assertTrue(cosine_sim_val.min().item() >= 0.9999)
            else:
                self.assertTrue(ref_out[idx] == ori_list[idx])
        del ref_out

        del a, b, c, d, a2
        del ori_list
        # check weak ref when some all objs are dead
        ref_out = [ref() for ref in weak_ref_list]
        self.assertTrue(ref_out[0] is None)
        self.assertTrue(ref_out[1] is None)
        self.assertTrue(ref_out[2] == 1.0)
        self.assertTrue(ref_out[3] == ["x", "y", "z"])

    def test_aclgraph_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

            def prompt(self, x):
                return self._forward(x)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

        model = Model()

        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([3, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x)  # cache hint
            model_match_cache(x)  # cache hint

    def test_aclgraph_cache_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config, dynamic=False)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config, dynamic=False)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(12, 12), True), [torch.ones(12, 12)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(*prompt2)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 12==3, stride 12==2 at dim=0", str(exception))

    def test_aclgraph_cache_dynamic_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(12, 2), True), [torch.ones(12, 2)]
        prompt2 = InputMeta(torch.ones(12, 12), True), [torch.ones(12, 12)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(*prompt2)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 12==12, stride 12==2 at dim=0", str(exception))

    def test_aclgraph_cache_capture_and_replay_keep_inference_input_mutations_true(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x):
                return self.cached_prompt(x)

            def prompt(self, x):
                return self._forward(x)

            def _forward(self, x):
                x.mul_(2)
                return x + 1

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        x_ = torch.randn([3, 2])
        x = x_.clone()

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x_)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        # inference
        with self.assertLogs(logger, level="WARNING") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("data_ptr is different between capture and replay." in log for log in cm.output),
            f"Expected WARNING 'Mutated input[arg]'s data_ptr is different between capture and replay.' "
            f"not found in logs: {cm.output}"
        )

    def test_compile_static_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        model = Model()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.aclgraph_config.kernel_aot_optimization = True
        config.aclgraph_config.kernel_aot_optimization_build_dir = ".."
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)
        x = torch.randn([3, 2])
        from torchair.core import _torchair
        _torchair.GetSocName()
        _torchair.AclopStartDumpArgs(1, "..")
        _torchair.AclopStopDumpArgs(1)

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model(x)
            except ValueError as e:
                messages = [str(w.message) for w in caught]
                self.assertTrue(
                    any("Starting static kernel compilation" in m for m in messages),
                    f"Expected warning 'Starting static kernel compilation' not found in {messages}"
                )

    def test_aclgraph_cache_npu_stream_switch_with_tagged_event(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, in1, in2, in3, in4):
                return self.cached_prompt(in1, in2, in3, in4)

            def prompt(self, in1, in2, in3, in4):
                return self._forward(in1, in2, in3, in4)


            def _forward(self, in1, in2, in3, in4):
                sub_result = torch.sub(in1, in2)
                torchair.ops.npu_tagged_event_record(ext_event1)
                torchair.ops.npu_tagged_event_record(ext_event2)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.npu_tagged_event_wait(ext_event1)
                    add_result1 = torch.add(in3, in4)
                    with torchair.scope.npu_stream_switch('1', 3):
                        torchair.ops.npu_tagged_event_wait(ext_event2)
                        add_result2 = torch.add(in3, in4)
                return sub_result, add_result1, add_result2


        ext_event1 = torchair.ops.npu_create_tagged_event(tag="6666")
        ext_event2 = torchair.ops.npu_create_tagged_event(tag="7777")

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        self.assertFalse(os.path.exists(prompt_cache_dir))
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        model(x, y, z, w)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    def test_aclgraph_cache_closure_vars(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x, y):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                x = x + y
                y = y + float('inf')
                empty = torch.ops.aten.empty([2, 2])
                return (x, y, empty)

            def prompt(self, x, y):
                return self._forward(x, y)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

        model = Model()

        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x, y)  # cache hint
            model_match_cache(x, y)  # cache hint

    def test_npu_multi_stream_with_multi_graph(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        ext_event1 = torchair.ops.npu_create_tagged_event(tag="666666")
        ext_event2 = torchair.ops.npu_create_tagged_event(tag="777777")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4, is_pr):
                add_result = torch.add(in1, in2)
                torchair.ops.npu_tagged_event_record(ext_event1)
                torchair.ops.npu_tagged_event_record(ext_event2)
                mm_result2 = add_result
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.npu_tagged_event_wait(ext_event1)
                    mm_result1 = torch.mm(in3, in4)
                    if is_pr:
                        with torchair.scope.npu_stream_switch('2', 3):
                            torchair.ops.npu_tagged_event_wait(ext_event2)
                            mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        from torchair._acl_concrete_graph.graph_pass import _GLOBAL_SCOPE_TAG_TO_EVENT
        opt_model(x, y, z, w, True)
        len_of_tagged_event_1 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        opt_model(x, y, z, w, False)
        len_of_tagged_event_2 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        opt_model(x, y, z, w, True)
        len_of_tagged_event_3 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        assert len_of_tagged_event_2 == len_of_tagged_event_3


if __name__ == '__main__':
    unittest.main()
