import contextlib
import logging
import os
import sys
import types
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair_st_utils import capture_stdout

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


# define stub aclgraph API
def stub_graph_pool_handle():
    logger.debug('[Stub] run stub API graph_pool_handle with args[].')
    pass


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
    def __new__(cls, device=None, priority=0, **kwargs):
        logger.debug('[Stub] new stub class Stream.')
        return "stream"

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        pass


def current_stream(device=None):
    logger.debug('[Stub] run stub API current_stream.')
    return "current_stream"


def current_device():
    logger.debug('[Stub] run stub API current_device.')
    return "current_device"
    # 无恢复操作


class StubExternalEvent:
    def __new__(cls):
        return super().__new__(cls)

    def record(self, stream=None):
        logger.debug('[Stub]ExternalEvent record.')
        return

    def wait(self, stream=None):
        logger.debug('[Stub]ExternalEvent wait.')
        return

    def reset(self, stream=None):
        logger.debug('[Stub]ExternalEvent reset.')
        return


# define stub submodule
class StubNpu:
    def __init__(self):
        logger.debug('[Stub] new stub module npu.')
        self.npu_fused_infer_attention_score = stub_fa
        self._npu_fused_infer_attention_score_get_max_workspace = stub_fa
        self.NPUGraph = StubNPUGraph
        self.graph = graph
        self.Stream = StubStream
        self.ExternalEvent = StubExternalEvent
        self.current_stream = current_stream
        self.current_device = current_device
        self.stream = stub_stream
        self.graph_pool_handle = stub_graph_pool_handle
        self.synchronize = stub_synchronize
        self.empty_cache = stub_empty_cache


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
    module.__all__ = ['npu']

    sys.modules['torch_npu'] = module
    logger.debug('[Stub] Original torch_npu.npu module is replaced by stub implementation: %s',
                 sys.modules['torch_npu'])
    return original_module


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
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
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
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
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
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        x = torch.randn([4, 3])
        model(x)
        first_model_id = id(model)
        x2 = torch.randn([5, 3])
        model(x2)
        second_model_id = id(model)
        self.assertTrue(first_model_id == second_model_id)

    def test_aclgraph_dynamic_sym_in_scale_and_tensor(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        def get_graph_num(concrete_graph):
            return len(concrete_graph.graph)

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
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        x = torch.randn([5, 2])
        scale1 = 4
        # torch._dynamo.reset()
        model(x, scale1)

        # no find captured graph, capture another npu graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 1, 1)
        with capture_stdout() as stdout:
            scale1 = 5
            model(x, scale1)
        captured_output = stdout.getvalue()
        self.assertTrue("Find captured acl graph for graph key " not in captured_output)

        # find captured graph, no need to capture another npu graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 2, 0)
        scale1 = 4
        model(x, scale1)

        # original fx graph no need to find captured graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 2, 1)
        scale1 = 4
        x2 = torch.randn([6, 2])
        model(x2, scale1)
        AclConcreteGraph.__call__ = bak_func

        # another fx graph no need to find captured graph
        AclConcreteGraph.__call__ = wrapper_call(bak_func, 0, 1)
        scale1 = 5.0
        model(x2, scale1)
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

    def test_aclgraph_capture_and_replay_with_multi_stream_external_event(self):
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_external_event(tag="11")

            def forward(self, x):
                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch1(xx):
                    y = xx + 1
                    y = y @ y
                    y = y * y
                    y = y - 1
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return y

                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch2(xx):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    torchair.ops.npu_tagged_event_reset(self.ext_event)
                    y = xx + 1
                    y = y @ y
                    y = y * y
                    y = y - 1
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

    def test_two_aclgraph_with_multi_stream_external_event(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        event_record_count = 0
        event_wait_count = 0
        event_reset_count = 0

        def get_graph_event_num(concrete_graph):
            nonlocal event_record_count
            nonlocal event_wait_count
            nonlocal event_reset_count
            for node in concrete_graph.fx_graph.graph.nodes:
                print(node.target)
                if node.name == "external_event_record":
                    event_record_count += 1
                if node.name == "external_event_wait":
                    event_wait_count += 1
                if node.name == "external_event_reset":
                    event_reset_count += 1

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0, f"expect len(args) > 0, but got {len(args)}"
                ret = func(*args, **kwargs)
                get_graph_event_num(args[0])
                return ret

            return wrapper

        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_external_event(tag="22")

            def forward(self, x):
                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch1(xx):
                    y = xx + 1
                    y = y @ y
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return y

                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch2(xx):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    torchair.ops.npu_tagged_event_reset(self.ext_event)
                    y = xx + 1
                    y = y * y
                    y = y @ y
                    return y

                with torch.npu.stream(self.stream1):
                    out1 = branch1(x)
                with torch.npu.stream(self.stream2):
                    out2 = branch2(x)
                return out1, out2

        model = Model()
        x = torch.randn([3, 3])
        model(x)

        assert event_record_count == 1, f"expect event record count is 1, but got {event_record_count}"
        assert event_wait_count == 1, f"expect event wait count is 1, but got {event_wait_count}"
        assert event_reset_count == 1, f"expect event reset count is 1, but got {event_reset_count}"

    def test_eager_with_multi_stream_external_event(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_external_event(tag="33")

            def forward(self, x):
                def branch1(xx):
                    y = xx + 1
                    y = y * y
                    y = y @ y
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return y

                def branch2(xx):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    torchair.ops.npu_tagged_event_reset(self.ext_event)
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

    def test_ge_with_multi_stream_external_event(self):
        from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
        # ignore event in ge mode
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ext_event = torchair.ops.npu_create_tagged_external_event(tag="44")

            def forward(self, xx):
                y = xx + xx
                y = y @ y
                y = y * y
                y = y + y
                torchair.ops.npu_tagged_event_record(self.ext_event)
                torchair.ops.npu_tagged_event_wait(self.ext_event)
                torchair.ops.npu_tagged_event_reset(self.ext_event)
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
            opt_model(x)

    def test_aclgraph_with_multi_stream_external_event_no_sync_device_called(self):
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stream1 = torch.npu.Stream()
                self.stream2 = torch.npu.Stream()
                self.ext_event = torchair.ops.npu_create_tagged_external_event(tag="55")

            def forward(self, x):
                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch1(xx):
                    y = xx + 1
                    y = y * y
                    y = y - 1
                    y = y @ y
                    torchair.ops.npu_tagged_event_record(self.ext_event)
                    return y

                @torch.compile(backend=aclgraph_backend, fullgraph=True, dynamic=False)
                def branch2(xx):
                    torchair.ops.npu_tagged_event_wait(self.ext_event)
                    torchair.ops.npu_tagged_event_reset(self.ext_event)
                    y = xx + 1
                    y = y @ y
                    return y

                with torch.npu.stream(self.stream1):
                    out1 = branch1(x)
                with torch.npu.stream(self.stream2):
                    out2 = branch2(x)
                return out1, out2

        self.stub_module.synchronize = Mock()  # mock torch.npu.synchronize, count sync call times
        model = Model()
        x = torch.randn([3, 3])
        for _ in range(2):
            model(x)
        # no sync called
        assert torch.npu.synchronize.call_count == 0, \
            f"expect no torch.npu.synchonize() call time is 0, but got {torch.npu.synchronize.call_count}"

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


if __name__ == '__main__':
    unittest.main()
