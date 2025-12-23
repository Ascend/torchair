import logging
import os
import sys
from typing import Tuple
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F
import _privateuse1_backend

import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
from torchair._acl_concrete_graph.utils import reconstruct_args_kwargs, WeakRef, LazyMessage

from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
    forbidden_attr,
    register_custom_ops,
)
from torchair_st_utils import capture_logger, generate_faked_module, register_is_npu


logger.setLevel(logging.DEBUG)


_get_pool_id = None

npu_device = _privateuse1_backend.npu_device()
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())


class AclGraphSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()
        register_custom_ops()
        register_is_npu()

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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
        config.experimental_config.keep_inference_input_mutations = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # inference
        with self.assertLogs(logger, level="DEBUG") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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

    def test_aclgraph_custom_update(self):
        from torchair._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
        _REPLACE_FUNC_MAP[torch.ops.custom.custom_infer.default] = StaticWorkspaceReplaceFunc(
            get_workspace=None,
            out_operator=torch.ops.custom.custom_infer.out,
            workspace_keys=[],
            output_keys=["out"],
            updated_param_keys=[],
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, dim):
                val = torch.ops.custom.custom_infer.default(x, y)
                x2 = x.sqrt()
                val2 = torch.ops.custom.custom_infer.default(x2, y)
                res = torch.cat([val, val2], dim=dim)
                return res

        model = Model()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([2, 3, 4, 5])
        y = torch.randn([2, 3, 4, 5])

        with capture_logger() as stdout:
            model(x, y, 0)
        self.assertTrue("Record all created sym expr and fx node" in stdout.getvalue())

        with capture_logger() as stdout:
            model(x, y, 1)
        self.assertTrue("Record all created sym expr and fx node" in stdout.getvalue())

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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            opt_model(x, x, x, x)

        self.assertTrue(
            any("tagged_event_record_default = torch.ops.air.tagged_event_record.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_record_default] "
            f"type[air.tagged_event_record.default]' in logs: {cm.output}")
        self.assertTrue(
            any("tagged_event_wait_default = torch.ops.air.tagged_event_wait.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        self.assertTrue(
            any("tagged_event_record_default_1 = torch.ops.air.tagged_event_record.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_1] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

        # 两条从流stream分别向主capture流发送record-wait对，以完成event闭环
        # stream tag 2
        self.assertTrue(
            any("tagged_event_wait_default_2 = torch.ops.air.tagged_event_wait.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_2] "
            "type[air.tagged_event_wait.default]' in logs: {cm.output}")

        # stream tag 1
        self.assertTrue(
            any("tagged_event_wait_default_3 = torch.ops.air.tagged_event_wait.default"
                in log for log in cm.output),
            f"Expected no DEBUG log 'Try to capture node names[tagged_event_wait_default_3] "
            f"type[air.tagged_event_wait.default]' in logs: {cm.output}")

    def test_npu_stream_switch_with_tagged_event(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
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

    def test_npu_stream_switch_with_super_kernel_scope(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                # superkernel scope no need to insert event wait
                with torchair.scope.super_kernel('1', "2"):
                    mm_result1 = torch.add(in3, in4)
                    # stream switch scope need to insert event wait
                    with torchair.scope.npu_stream_switch('2', 3):
                        mm_result2 = torch.mm(in3, in4)
                # limit core num scope no need to insert event wait
                with torchair.scope.limit_core_num(1, 2):
                    mm_result3 = torch.add(in3, in4)
                return add_result, mm_result1, mm_result2, mm_result3

        def check_graph(concrete_graph):
            event_record = 0
            for node in concrete_graph.fx_graph.graph.nodes:
                if str(node.target) == "aten.mm.default":
                    assert str(node.prev.target) == "air.tagged_event_wait.default"
                if str(node.target) == "air.tagged_event_record.default":
                    event_record += 1
            assert event_record == 2, f"expect event record count is 2, but got {event_record}"

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) >= 3
                check_graph(args[0])
                return tuple([args[0], args[1], args[2], args[3]])

            return wrapper

        AclConcreteGraph.__call__ = decorator(AclConcreteGraph.__call__)

        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        with capture_logger() as stdout:
            try:
                opt_model(x, y, z, w)
            except Exception:
                pass
        self.assertTrue("current_stream = torchair_st_stub_aclgraph_utils_current_stream()" in stdout.getvalue())




    def test_npu_stream_switch_with_super_kernel_scope_with_nest_scope(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                # superkernel scope no need to insert event wait
                with torchair.scope.super_kernel('1', "2"):
                    mm_result1 = torch.add(in3, in4)
                    # stream switch scope need to insert event wait
                    with torchair.scope.npu_stream_switch('2', 3):
                        mm_result2 = torch.mm(in3, in4)
                        # limit core num scope no need to insert event wait
                        with torchair.scope.limit_core_num(1, 2):
                            mm_result3 = torch.add(in3, in4)
                            with torchair.scope.super_kernel('1', "2"):
                                mm_result4 = torch.add(in3, in4)
                return add_result, mm_result1, mm_result2, mm_result3, mm_result4

        def check_graph(concrete_graph):
            event_record = 0
            for node in concrete_graph.fx_graph.graph.nodes:
                if str(node.target) == "aten.mm.default":
                    assert str(node.prev.target) == "air.tagged_event_wait.default"
                if str(node.target) == "air.tagged_event_record.default":
                    event_record += 1
            assert event_record == 2, f"expect event record count is 2, but got {event_record}"

        def decorator(call):
            def wrapper(*args, **kwargs):
                assert len(args) >= 3
                check_graph(args[0])
                return tuple([args[0], args[1], args[2], args[3], args[4]])

            return wrapper

        AclConcreteGraph.__call__ = decorator(AclConcreteGraph.__call__)

        model = Model()
        opt_model = torch.compile(model, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])
        with capture_logger() as stdout:
            try:
                opt_model(x, y, z, w)
            except Exception:
                pass
        self.assertTrue("current_stream = torchair_st_stub_aclgraph_utils_current_stream()" in stdout.getvalue())

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
        config_view.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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

        self.assertTrue(
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
        config.debug.aclgraph.clone_input = False
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.remove_noop_ops = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
        config.experimental_config.keep_inference_input_mutations = True
        config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        config.experimental_config.remove_noop_ops = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
        config.experimental_config.keep_inference_input_mutations = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = False
        config.experimental_config.remove_noop_ops = False
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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
                # shape of viewed tensor may be changed after reconstructing outputs
                # will test view of outputs after forward fixing
                return ln1

        torch._dynamo.reset()
        x = torch.randn([4, 2])
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        # when only one graph, no need reconstruct
        model1 = Model()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res1 = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("When mempool reuse is enabled in fx_graph" in captured_output)

        # second graph with valid output ref, need reconstruct
        y = torch.randn([5, 2])
        with capture_logger() as stdout:
            res1 = model1(y)
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)  # should be true in real env

        # same graph with valid output ref, no need reconstruct
        with capture_logger() as stdout:
            res1 = model1(y)
        captured_output = stdout.getvalue()
        self.assertTrue("no need to reconstruct output tensors for" in captured_output)  # should be true in real env

        # same graph with invalid output ref, need reconstruct
        del res1
        with capture_logger() as stdout:
            res1 = model1(y)
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
        config.debug.aclgraph.clone_input = False
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
        y = torch.randn([4, 2])
        torch._dynamo.reset()

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        from torchair._acl_concrete_graph.acl_graph import AclGraph

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)

                assert len(args) > 0
                acl_graph = args[0]
                global _get_pool_id
                _get_pool_id = acl_graph.pool
                return ret

            return wrapper

        AclGraph.__call__ = wrapper_call(AclGraph.__call__)

        # test no set custom pool, check different pool, and reconstruct outputs
        model1 = Model1()
        model1 = torch.compile(model1, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("When mempool reuse is enabled in" in captured_output)

        with capture_logger() as stdout:
            res = model1(y)
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)
        pool_id1 = _get_pool_id

        model2 = Model2()
        model2 = torch.compile(model2, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            res = model2(x)
        captured_output = stdout.getvalue()
        self.assertTrue("When mempool reuse is enabled in" in captured_output)

        with capture_logger() as stdout:
            res = model2(y)
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

    def test_lazy_message(self):
        mock_func1 = Mock(return_value="test_func1")
        lazy_message = LazyMessage(mock_func1, "arg1", "arg2")
        logger.debug("Debug message : %s", lazy_message)
        mock_func1.assert_called_once_with("arg1", "arg2")

        mock_func2 = Mock(return_value="test_func2")
        logger.setLevel(logging.INFO)
        lazy_message = LazyMessage(mock_func2, "arg1", "arg2")
        logger.debug("Debug message : %s", lazy_message)
        mock_func2.assert_not_called()

        logger.setLevel(logging.DEBUG)

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
        config.debug.aclgraph.clone_input = False
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
                self.cached = torchair.inference.cache_compile(self._forward, config=config, dynamic=False)

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)


        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model._forward, config=config, dynamic=False)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]

        t12 = torch.zeros(1, 5)
        t22 = torch.zeros(2, 5)[:, 0:1]
        t32 = torch.zeros(5, 2)
        s12 = 5
        s22 = [2, 3]
        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(t1, t2, t3, s1, s2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(t12, t22, t32, s12, s22)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 5==10, stride 1==1 at dim=1", str(exception))

    def test_aclgraph_cache_dynamic_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = torchair.inference.cache_compile(self._forward, config=config)

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)


        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model._forward, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]

        t12 = torch.zeros(2, 5)
        t22 = torch.zeros(2, 5)[:, 0:1]
        t32 = torch.zeros(5, 2)
        s12 = 5
        s22 = [2, 3]
        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(t1, t2, t3, s1, s2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(t12, t22, t32, s12, s22)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 2==1, stride 5==5 at dim=0", str(exception))

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
        config.debug.aclgraph.clone_input = False
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
        with self.assertLogs(logger, level="DEBUG") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured' "
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
        config.debug.aclgraph.clone_input = False
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = ".."
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
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        self.assertFalse(os.path.exists(prompt_cache_dir))
        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])

        model = Model()
        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x, y, z, w)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        self.assertTrue(
            any("with torch.npu.stream" in log for log in cm.output),
            f"Expected fx_forward DEBUG log 'with torch.npu.stream'"
            f"not found in logs: {cm.output}"
        )
        model2 = Model()
        model2(x, y, z, w)
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
        config.debug.aclgraph.clone_input = False
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
        config.debug.aclgraph.clone_input = False
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
        from torchair.scope._scope_attr import _GLOBAL_TAG_TO_STREAM
        opt_model(x, y, z, w, True)
        len_of_tagged_event_1 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_1 = len(_GLOBAL_TAG_TO_STREAM)
        opt_model(x, y, z, w, False)
        len_of_tagged_event_2 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_2 = len(_GLOBAL_TAG_TO_STREAM)
        opt_model(x, y, z, w, True)
        len_of_tagged_event_3 = len(_GLOBAL_SCOPE_TAG_TO_EVENT)
        len_of_stream_3 = len(_GLOBAL_TAG_TO_STREAM)
        assert len_of_tagged_event_2 == len_of_tagged_event_3
        assert len_of_stream_2 == len_of_stream_3

    def test_aclgraph_supported_blocking_env(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + x

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=aclgraph_backend, dynamic=False)

        os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
        with capture_logger() as stdout:
            model(torch.randn([4, 2]))
        self.assertIn("Success to capture fx_graph", stdout.getvalue())

        with capture_logger() as stdout:
            model(torch.randn([5, 2]))
        self.assertIn("Success to capture fx_graph", stdout.getvalue())
        os.environ['ASCEND_LAUNCH_BLOCKING'] = '0'

    def test_aclgraph_static_capture_size_limit(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, input):
                return self.linear(input)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        # test use default static_capture_size_limit, and do not fall back to eager
        torch._dynamo.reset()
        model1 = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([4, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([5, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        # test set custom static_capture_size_limit, and fall back to eager
        config.debug.aclgraph.static_capture_size_limit = 1
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        torch._dynamo.reset()
        model1 = torch.compile(Model(), backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([4, 2]))
        self.assertTrue("static_capture_size_limit reached" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        # fall back to eager no aclgraph log
        self.assertTrue("Find captured AclGraph" not in stdout.getvalue())
        self.assertTrue("Success to capture fx_graph" not in stdout.getvalue())

    def test_aclgraph_static_capture_size_limit_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, input):
                return self.cached_prompt(input)

            def prompt(self, input):
                return input + input

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.static_capture_size_limit = 1
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model1 = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model1.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        self.assertFalse(os.path.exists(prompt_cache_dir))

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([4, 2]))
        self.assertTrue("static_capture_size_limit reached" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        # fall back to eager no aclgraph log
        self.assertTrue("Find captured AclGraph" not in stdout.getvalue())
        self.assertTrue("Success to capture fx_graph" not in stdout.getvalue())

        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    def test_aclgraph_recapture_non_mutated_input_with_address_change_and_input_with_no_address_change(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        model = Model()
        config = CompilerConfig()
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        model(x_)
        with self.assertLogs(logger, level="DEBUG") as cm:
            output = model(x)

        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

        with capture_logger() as stdout:
            model(x)
        captured_output = stdout.getvalue()
        self.assertTrue("Find captured AclGraph" in captured_output)
        self.assertTrue("The current AclGraph no needs to be recaptured" in captured_output)

    def test_aclgraph_recapture_mutated_input_with_address_change(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.mul_(2)
                return x + 1

        model = Model()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x_ = torch.randn([3, 2])
        x = x_.clone()
        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x_)
            output = model(x)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_recapture_multi_graph_with_address_change(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.mul_(2)
                return x + 1

        model = Model()
        config = CompilerConfig()
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.ones([3, 2])
        y = torch.ones([4, 2])
        z = torch.ones([2, 1])
        x_ = x.clone()

        with capture_logger() as stdout:
            model(x)
            model(y)
            model(z)
        captured_output = stdout.getvalue()
        self.assertTrue("No find captured AclGraph" in captured_output)

        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x_)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_recapture_with_parameter_address_change(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        config = CompilerConfig()
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])
        a = torch.ones(2, 2)
        model.linear.weight.data = a
        model(x)

        b = torch.zeros(2, 2)
        model.linear.weight.data = b
        with capture_logger() as stdout:
            model(x)
        captured_output = stdout.getvalue()
        self.assertFalse("The current AclGraph needs to be recaptured" in captured_output)

        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_recapture_with_parameter_address_change_with_frozen_parameter(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])
        a = torch.ones(2, 2)
        model.linear.weight.data = a

        model(x)
        b = torch.zeros(2, 2)
        model.linear.weight.data = b

        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_cache_tensor_constant(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x, y):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                x = x + y
                x = torch.maximum(x, torch.tensor(torch.finfo(x.dtype).min, device=x.device))
                return x

            def prompt(self, x, y):
                return self._forward(x, y)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
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

    def test_aclgraph_cache_compile_with_static_kernel(self):
        class StaticKernelModel(torch.nn.Module):
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
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = ".."
        model = StaticKernelModel()

        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        config.experimental_config.aclgraph._aclnn_static_shape_kernel = False
        model2 = StaticKernelModel()
        prompt2_cache_bin = CompiledModel.get_cache_bin(model2.prompt, config=config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt2_cache_bin)))
        prompt2_cache_dir = os.path.abspath(os.path.dirname(prompt2_cache_bin))
        self.assertNotEqual(prompt2_cache_dir, prompt_cache_dir,
                            "Cache bin dir with different config should not be the same.")

        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        model3 = StaticKernelModel()
        prompt3_cache_bin = CompiledModel.get_cache_bin(model3.prompt, config=config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt3_cache_bin)))
        prompt3_cache_dir = os.path.abspath(os.path.dirname(prompt3_cache_bin))
        self.assertEqual(prompt3_cache_dir, prompt_cache_dir,
                            "Cache bin dir with same config and same model should be the same.")

    def test_npu_stream_record_wait_with_wait(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                mm = torch.mm(x, x)
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.wait([mm])
                    abs1 = torch.abs(mm)
                    add1 = torch.add(abs1, 1)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([mm, abs1])
                    sub = torch.sub(add1, mm)
                return sub

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = Model()
        opt_model = torch.compile(model, backend=npu_backend, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x)

        self.assertTrue(
            any("record_default = torch.ops.air.record.default()"
                    in log for log in cm.output),
        f"Expected DEBUG log 'record_default = torch.ops.air.record.default()' in logs: {cm.output}")
        self.assertTrue(
            any("wait = torch.ops.air.wait.default([record_default])"
                    in log for log in cm.output),
        f"Expected DEBUG log 'wait = torch.ops.air.wait.default([record_default])' in logs: {cm.output}")
        self.assertTrue(
            any("End insert record node in graph"
                    in log for log in cm.output),
        f"Expected DEBUG log 'End insert record node in graph' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_record.default("
                in log for log in cm.output),
            f"Expected DEBUG log 'Record successfully,stream:' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_wait.default("
                in log for log in cm.output),
            f"Expected DEBUG log 'Wait successfully,stream:' in logs: {cm.output}")
    
    def test_npu_stream_record_wait_with_record(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                mm = torch.mm(x, x)
                result1 = torchair.ops.record()
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.wait([result1])
                    abs1 = torch.abs(mm)
                    result2 = torchair.ops.record()
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([result2])
                    add1 = torch.add(abs1, 1)
                    result3 = torchair.ops.record()
                with torchair.scope.npu_stream_switch('3', 3):
                    torchair.ops.wait([result2, result3])
                    sub = torch.sub(abs1, add1)
                return sub

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = Model()
        opt_model = torch.compile(model, backend=npu_backend, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x)

        self.assertTrue(
            any("record = torch.ops.air.record.default()"
                    in log for log in cm.output),
        f"Expected DEBUG log 'record = torch.ops.air.record.default()' in logs: {cm.output}")
        self.assertTrue(
            any("wait = torch.ops.air.wait.default([record])"
                    in log for log in cm.output),
        f"Expected DEBUG log 'wait = torch.ops.air.wait.default([record])' in logs: {cm.output}")
        self.assertTrue(
            any("End insert record node in graph"
                    in log for log in cm.output),
        f"Expected DEBUG log 'End insert record node in graph' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_record.default("
                    in log for log in cm.output),
        f"Expected DEBUG log 'Record successfully,stream:' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_wait.default("
                    in log for log in cm.output),
        f"Expected DEBUG log 'Wait successfully,stream:' in logs: {cm.output}")

    def test_npu_stream_record_wait_with_record_wait(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                mm = torch.mm(x, x)
                result1 = torchair.ops.record()
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.wait([result1])
                    abs1 = torch.abs(mm)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([abs1])
                    add1 = torch.add(abs1, 1)
                    result3 = torchair.ops.record()
                with torchair.scope.npu_stream_switch('3', 3):
                    torchair.ops.wait([abs1, result3])
                    sub = torch.sub(abs1, add1)
                return sub

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = Model()
        opt_model = torch.compile(model, backend=npu_backend, dynamic=False)
        x = torch.randn([3, 3])
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x)

        self.assertTrue(
            any("record_default = torch.ops.air.record.default()"
                    in log for log in cm.output),
        f"Expected DEBUG log 'record_default = torch.ops.air.record.default()' in logs: {cm.output}")
        self.assertTrue(
            any("wait_1 = torch.ops.air.wait.default([record_default])"
                    in log for log in cm.output),
        f"Expected DEBUG log 'wait_1 = torch.ops.air.wait.default([record_default])' in logs: {cm.output}")
        self.assertTrue(
            any("End insert record node in graph"
                    in log for log in cm.output),
        f"Expected DEBUG log 'End insert record node in graph' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_record.default("
                    in log for log in cm.output),
        f"Expected DEBUG log 'Record successfully,stream:' in logs: {cm.output}")
        self.assertTrue(
            any("torch.ops.air.tagged_event_wait.default("
                    in log for log in cm.output),
        f"Expected DEBUG log 'Wait successfully,stream:' in logs: {cm.output}")

    def test_aclgraph_cache_with_record_wait(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)

            def forward(self, x):
                return self.cached_prompt(x)

            def prompt(self, x):
                return self._forward(x)

            def _forward(self, x):
                mm = torch.mm(x, x)
                with torchair.scope.npu_stream_switch('1', 3):
                    torchair.ops.wait([mm])
                    abs1 = torch.abs(mm)
                    add1 = torch.add(abs1, 1)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([mm, abs1])
                    sub = torch.sub(add1, mm)
                return sub

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=config)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        self.assertFalse(os.path.exists(prompt_cache_dir))
        x = torch.randn([3, 3])
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_multi_mutated_input(self):
        def f(x, out_sin, out_cos):
            return torch.ops.custom.sin_cos_inplace.default(x, out_sin, out_cos)

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        x = torch.randn(3)
        sin = torch.randn(3)
        cos = torch.randn(3)
        model = torch.compile(f, backend=aclgraph_backend)
        with self.assertLogs(logger, level="DEBUG") as cm:
            res = model(x, sin, cos)
        self.assertTrue(
            any("call_function[target=torch.ops.custom.sin_cos_inplace.default]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.custom.sin_cos_inplace.default]' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_multi_mutated_input_with_view_before(self):
        def f(x, out_sin, out_cos):
            sin_view = out_sin.view(-1, 1)
            y = torch.ops.custom.sin_cos_inplace.default(x, sin_view, out_cos)
            res = out_cos + 1
            return y, res

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        x = torch.ones(3)
        sin = torch.ones(3)
        cos = torch.ones(3)
        model = torch.compile(f, backend=aclgraph_backend)
        with self.assertLogs(logger, level="DEBUG") as cm:
            res = model(x, sin, cos)
        self.assertTrue(
            any("reinplace failed" in log for log in cm.output),
            f"Expected DEBUG log 'reinplace failed' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    def test_multi_mutated_input_with_view_after(self):
        def f(x, out_sin, out_cos):
            y = torch.ops.custom.sin_cos_inplace.default(x, out_sin, out_cos)
            sin_view = out_sin.view(-1, 1)
            res = sin_view + 1
            return y, res

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)
        x = torch.ones(3)
        sin = torch.ones(3)
        cos = torch.ones(3)
        model = torch.compile(f, backend=aclgraph_backend)
        with self.assertLogs(logger, level="DEBUG") as cm:
            res = model(x, sin, cos)
        self.assertTrue(
            any("call_function[target=torch.ops.custom.sin_cos_inplace.default]" in log for log in cm.output),
            f"Expected DEBUG log 'call_function[target=torch.ops.custom.sin_cos_inplace.default]' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_slice_acl(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                slice_x = x[:]
                slice_y = y[:]
                return slice_x + slice_y

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_slice_node = any(node.op == "call_function"
                                 and node.target.overloadpacket == torch.ops.aten.slice for node in nodes)
            assert not has_slice_node

        def wrapper_call(call):
            def wrapper(*args, **kwargs):
                ret = call(*args, **kwargs)
                assert_func(args[0])
                return ret

            return wrapper

        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        call_bak = AclConcreteGraph.__call__
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        try:
            config_ = CompilerConfig()
            config_.mode = "reduce-overhead"
            config_.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            backend = torchair.get_npu_backend(compiler_config=config_)
            compiled_model = torch.compile(Model(), backend=backend, dynamic=True)
            _ = compiled_model(x=torch.randn([2, 2]), y=torch.randn([2, 2]))
        finally:
            AclConcreteGraph.__call__ = call_bak

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_enable_remove_noop_ops_and_eliminate_dead_code_acl(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                zeros_tensor = torch.zeros_like(x)
                zeros_copy = zeros_tensor.copy_(x)
                res = zeros_copy + 1
                return res

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_zeros_like_node = any(node.op == "call_function"
                                 and node.target.overloadpacket == torch.ops.aten.zeros_like for node in nodes)
            assert not has_zeros_like_node

        def wrapper_call(call):
            def wrapper(*args, **kwargs):
                ret = call(*args, **kwargs)
                assert_func(args[0])
                return ret

            return wrapper

        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        compiler_config.debug.aclgraph.clone_input = False
        backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(Model(), backend=backend, dynamic=True)
        compiled_model(x=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < '2.2.0', "")
    def test_disable_remove_noop_ops_and_eliminate_dead_code_acl(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                zeros_tensor = torch.zeros_like(x)
                zeros_copy = zeros_tensor.copy_(x)
                res = zeros_copy + 2
                return res

        def assert_func(concrete_graph):
            graph_ = concrete_graph.fx_graph.graph
            nodes = graph_.nodes
            has_zeros_like_node = any(node.op == "call_function"
                                and node.target.overloadpacket == torch.ops.aten.zeros_like for node in nodes)
            assert has_zeros_like_node

        def wrapper_call(call):
            def wrapper(*args, **kwargs):
                ret = call(*args, **kwargs)
                assert_func(args[0])
                return ret

            return wrapper

        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        compiler_config.debug.aclgraph.clone_input = False
        compiler_config.experimental_config.remove_noop_ops = False
        backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(Model(), backend=backend, dynamic=True)
        compiled_model(x=torch.randn([2, 2]))

    @unittest.skipIf(torch.__version__ < "2.6", "")
    def test_torch_compile_acl_debug_dump(self):

        # Define templates for ACL mode (without hardcoded indices)
        ACL_COMMON_FILES = [
            "dynamo_out_graph.txt",
        ]

        ACL_FORWARD_STEP_TEMPLATS = [
            "aot_forward_graph.txt",
            "aot_forward_graph_after_post_grad_custom_pre_pass.txt",
            "aot_forward_graph_after_optimize_noop_ops.txt",
            "aot_forward_graph_after_recover_view_inplace_pattern.txt",
            "aot_forward_graph_after_apply_pattern_passes.txt",
            "aot_forward_graph_after_view_to_reshape.txt",
            "aot_forward_graph_after_post_grad_custom_post_pass.txt",
            "aot_forward_graph_after_apply_event_closure_with_multi_stream.txt",
            "aot_forward_graph_after_apply_event_record.txt",
            "aot_forward_graph_after_eliminate_dead_code.txt",
            "aot_forward_graph_after_reinplace_inplaceable_ops_pass.txt",
            "aot_forward_graph_after_reinplace_input_mutated_ops.txt",
            "aot_forward_graph_after_decompose_auto_functionalized.txt",
            "aot_forward_graph_after_replace_dynamic_workspace_ops.txt",
            "aot_forward_graph_after_replace_core_limit_nodes.txt",
        ]

        ACL_BACKWARD_STEP_TEMPLATES = [
            "aot_backward_graph.txt",
            "aot_backward_graph_after_post_grad_custom_pre_pass.txt",
            "aot_backward_graph_after_optimize_noop_ops.txt",
            "aot_backward_graph_after_recover_view_inplace_pattern.txt",
            "aot_backward_graph_after_apply_pattern_passes.txt",
            "aot_backward_graph_after_view_to_reshape.txt",
            "aot_backward_graph_after_post_grad_custom_post_pass.txt",
            "aot_backward_graph_after_apply_event_closure_with_multi_stream.txt",
            "aot_backward_graph_after_apply_event_record.txt",
            "aot_backward_graph_after_eliminate_dead_code.txt",
            "aot_backward_graph_after_reinplace_inplaceable_ops_pass.txt",
            "aot_backward_graph_after_reinplace_input_mutated_ops.txt",
            "aot_backward_graph_after_decompose_auto_functionalized.txt",
            "aot_backward_graph_after_replace_dynamic_workspace_ops.txt",
            "aot_backward_graph_after_replace_core_limit_nodes.txt",
        ]

        # Generate patterns with auto-incremented indics
        def generate_acl_patterns():
            patterns = []
            # Add common files
            for file in ACL_COMMON_FILES:
                patterns.append(f"model__{{id}}/{file}")
            # Add forward files with auto indices
            for idx, template in enumerate(ACL_FORWARD_STEP_TEMPLATS):
                patterns.append(f"model__{{id}}/forward/{idx:03d}_{template}")
            patterns.append(f"model__{{id}}/forward/output_code.py")
            for idx, template in enumerate(ACL_BACKWARD_STEP_TEMPLATES):
                patterns.append(f"model__{{id}}/backward/{idx:03d}_{template}")
            patterns.append(f"model__{{id}}/backward/output_code.py")
            return patterns

        EXPECTED_FILE_PATTERNS_ACL = generate_acl_patterns()


        from torch._dynamo.utils import get_debug_dir
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchair_debug_") as tmpdir:
            DEBUG_DIR = tmpdir
            torch._dynamo.config.patch(debug_dir_root=DEBUG_DIR)
            self.assertIsNotNone(DEBUG_DIR)
            os.environ['TORCH_COMPILE_DEBUG'] = '1'

            config = torchair.CompilerConfig()
            config.experimental_config.remove_noop_ops = True
            config.mode.value = "reduce-overhead"

            def _custom_pre_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
                return None

            def _custom_post_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
                return None

            config.post_grad_custom_pre_pass = _custom_pre_fn
            config.post_grad_custom_post_pass = _custom_post_fn
            npu_backend = torchair.get_npu_backend(compiler_config=config)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return 2 * x

            model = Model()
            compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
            x = torch.randn(10, 10, requires_grad=True)
            out = compiled_model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.randn(10, 10)
            loss = loss_fn(out, target)
            loss.backward()

            debug_dir_output = get_debug_dir()
            # 1. Verify the existence of the torchair directory
            torchair_root = os.path.join(debug_dir_output, "torchair")
            self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

            # 2. Verify all expected files exist
            expected_files = []
            for template in EXPECTED_FILE_PATTERNS_ACL:
                rel_path = template.format(id=0)
                expected_files.append(rel_path)

            def check_torchair_directory_structure(base_dir: str, file_list: list) -> list:
                missing_files = []
                for rel_path in file_list:
                    abs_path = os.path.join(base_dir, rel_path)
                    if not os.path.exists(abs_path):
                        missing_files.append(abs_path)
                return missing_files
            missing_files = check_torchair_directory_structure(torchair_root, expected_files)
            self.assertFalse(missing_files, msg=f"Missing files: {', '.join(missing_files)}")

            # Check file count to ensure no extra files
            expected_count = len(EXPECTED_FILE_PATTERNS_ACL)
            actual_count = 0
            for root, _, files in os.walk(torchair_root):
                actual_count += len(files)
            self.assertEqual(
                actual_count,
                expected_count,
                msg=f"File count mismatch: expected {expected_count} files, got {actual_count} files"
            )

    @unittest.skipIf(torch.__version__ > "2.2", "")
    def test_torch_compile_acl_debug_dump_low_version(self):

        # Define templates for ACL mode (without hardcoded indices)
        ACL_COMMON_FILES = [
            "dynamo_out_graph.txt",
        ]

        ACL_FORWARD_STEP_TEMPLATS = [
            "aot_forward_graph.txt",
            "aot_forward_graph_after_post_grad_custom_pre_pass.txt",
            "aot_forward_graph_after_optimize_noop_ops.txt",
            "aot_forward_graph_after_recover_view_inplace_pattern.txt",
            "aot_forward_graph_after_apply_pattern_passes.txt",
            "aot_forward_graph_after_view_to_reshape.txt",
            "aot_forward_graph_after_post_grad_custom_post_pass.txt",
            "aot_forward_graph_after_apply_event_closure_with_multi_stream.txt",
            "aot_forward_graph_after_apply_event_record.txt",
            "aot_forward_graph_after_eliminate_dead_code.txt",
            "aot_forward_graph_after_reinplace_input_mutated_ops.txt",
            "aot_forward_graph_after_replace_dynamic_workspace_ops.txt",
            "aot_forward_graph_after_replace_core_limit_nodes.txt",
        ]

        ACL_BACKWARD_STEP_TEMPLATES = [
            "aot_backward_graph.txt",
            "aot_backward_graph_after_post_grad_custom_pre_pass.txt",
            "aot_backward_graph_after_optimize_noop_ops.txt",
            "aot_backward_graph_after_recover_view_inplace_pattern.txt",
            "aot_backward_graph_after_apply_pattern_passes.txt",
            "aot_backward_graph_after_view_to_reshape.txt",
            "aot_backward_graph_after_post_grad_custom_post_pass.txt",
            "aot_backward_graph_after_apply_event_closure_with_multi_stream.txt",
            "aot_backward_graph_after_apply_event_record.txt",
            "aot_backward_graph_after_eliminate_dead_code.txt",
            "aot_backward_graph_after_reinplace_input_mutated_ops.txt",
            "aot_backward_graph_after_replace_dynamic_workspace_ops.txt",
            "aot_backward_graph_after_replace_core_limit_nodes.txt",
        ]

        # Generate patterns with auto-incremented indics
        def generate_acl_patterns():
            patterns = []
            # Add common files
            for file in ACL_COMMON_FILES:
                patterns.append(f"model__{{id}}/{file}")
            # Add forward files with auto indices
            for idx, template in enumerate(ACL_FORWARD_STEP_TEMPLATS):
                patterns.append(f"model__{{id}}/forward/{idx:03d}_{template}")
            patterns.append(f"model__{{id}}/forward/output_code.py")
            for idx, template in enumerate(ACL_BACKWARD_STEP_TEMPLATES):
                patterns.append(f"model__{{id}}/backward/{idx:03d}_{template}")
            patterns.append(f"model__{{id}}/backward/output_code.py")
            return patterns

        EXPECTED_FILE_PATTERNS_ACL = generate_acl_patterns()


        from torch._dynamo.utils import get_debug_dir
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchair_debug_") as tmpdir:
            DEBUG_DIR = tmpdir
            torch._dynamo.config.patch(debug_dir_root=DEBUG_DIR)
            self.assertIsNotNone(DEBUG_DIR)
            os.environ['TORCH_COMPILE_DEBUG'] = '1'

            config = torchair.CompilerConfig()
            config.experimental_config.remove_noop_ops = True
            config.mode.value = "reduce-overhead"
            config.debug.aclgraph.clone_input = False

            def _custom_pre_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
                return None

            def _custom_post_fn(gm, example_inputs, compile_config: torchair.CompilerConfig):
                return None

            config.post_grad_custom_pre_pass = _custom_pre_fn
            config.post_grad_custom_post_pass = _custom_post_fn
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            npu_backend = torchair.get_npu_backend(compiler_config=config)

            class Model(torch.nn.Module):
                def forward(self, x):
                    return 2 * x

            model = Model()
            compiled_model = torch.compile(model, backend=npu_backend, dynamic=False)
            x = torch.randn(10, 10, requires_grad=True)
            out = compiled_model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.randn(10, 10)
            loss = loss_fn(out, target)
            loss.backward()

            debug_dir_output = get_debug_dir()
            # 1. Verify the existence of the torchair directory
            torchair_root = os.path.join(debug_dir_output, "torchair")
            self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

            # 2. Verify all expected files exist
            expected_files = []
            for template in EXPECTED_FILE_PATTERNS_ACL:
                rel_path = template.format(id=0)
                expected_files.append(rel_path)

            def check_torchair_directory_structure(base_dir: str, file_list: list) -> list:
                missing_files = []
                for rel_path in file_list:
                    abs_path = os.path.join(base_dir, rel_path)
                    if not os.path.exists(abs_path):
                        missing_files.append(abs_path)
                return missing_files
            missing_files = check_torchair_directory_structure(torchair_root, expected_files)
            self.assertFalse(missing_files, msg=f"Missing files: {', '.join(missing_files)}")

            # Check file count to ensure no extra files
            expected_count = len(EXPECTED_FILE_PATTERNS_ACL)
            actual_count = 0
            for root, _, files in os.walk(torchair_root):
                actual_count += len(files)
            self.assertEqual(
                actual_count,
                expected_count,
                msg=f"File count mismatch: expected {expected_count} files, got {actual_count} files"
            )

    def test_limit_core_num_in_aclgraph(self):
        from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.limit_core_num(2, 4):
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.limit_core_num(12, 24):
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.limit_core_num(2, 4):
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.npu_stream_switch('1', 3):
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        class Model3(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.limit_core_num(2, 4):
                    mm_result1 = torch.mm(in3, in4)
                    with torchair.scope.super_kernel('1', '2'):
                        mm_result2 = torch.mm(in3, in4)
                return add_result, mm_result1, mm_result2

        def check_graph(concrete_graph):
            scope_enter_count = 0
            set_stream_count = 0
            get_stream_count = 0
            for node in concrete_graph.fx_graph.graph.nodes:
                if str(node.target) == "air.scope_enter.default":
                    scope_enter_count += 1

                if str(node.target) == "air.scope_exit.default":
                    scope_enter_count -= 1
                if "function get_stream_limit" in str(node.target):
                    assert "function current_stream" in str(node.prev.target)
                    assert str(node.next.target) == str(node.next.next.target)
                    assert "function set_stream_limit" in str(node.next.next.next.target)
                    get_stream_count += 1
                if "function set_stream_limit" in str(node.target):
                    set_stream_count += 1

            assert scope_enter_count == 0, f"expect scope enter should match with scope exit, but got unmatched"
            assert set_stream_count == 2 * get_stream_count

        def wrapper_call(call):
            def wrapper(*args, **kwargs):
                assert len(args) >= 3
                ret = call(*args, **kwargs)
                check_graph(args[0])
                return ret

            return wrapper

        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        x = torch.randn([3, 3])
        y = torch.randn([3, 3])
        z = torch.randn([3, 3])
        w = torch.randn([3, 3])

        model1 = Model1()
        model1 = torch.compile(model1, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        with capture_logger() as stdout:
            try:
                model1(x, y, z, w)
            except Exception:
                pass
        self.assertTrue("current_stream = torchair_st_stub_aclgraph_utils_current_stream()" in stdout.getvalue())
        self.assertTrue("Codegen for graph" in stdout.getvalue())
        
        model2 = Model2()
        model2 = torch.compile(model2, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        with capture_logger() as stdout:
            try:
                model2(x, y, z, w)
            except Exception:
                pass
        self.assertTrue("current_stream = torchair_st_stub_aclgraph_utils_current_stream()" in stdout.getvalue())
        self.assertTrue("Codegen for graph" in stdout.getvalue())
        
        model3 = Model3()
        model3 = torch.compile(model3, backend=aclgraph_backend, fullgraph=True, dynamic=False)
        with capture_logger() as stdout:
            try:
                model3(x, y, z, w)
            except Exception:
                pass
        self.assertTrue("current_stream = torchair_st_stub_aclgraph_utils_current_stream()" in stdout.getvalue())
        self.assertTrue("Codegen for graph" in stdout.getvalue())

    def test_aclgraph_userinput_construct_in_share_memory_with_parameter_and_mutated(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, y):
                x.mul_(2)
                x = x @ self.linear(y)
                return x + 1

        torch._dynamo.reset()
        x = torch.ones([3, 2])
        y = torch.ones([4, 2])
        a = torch.ones(2, 2)
        b = torch.zeros(2, 2)
        c = torch.ones([2, 2])
        x_ = x.clone()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.experimental_config.keep_inference_input_mutations = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        # when only one graph, no need reconstruct
        model = Model()
        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        with capture_logger() as stdout:
            model.linear.weight.data = a
            res1 = model(x, c)
        captured_output = stdout.getvalue()
        self.assertTrue("When mempool reuse is enabled in fx_graph" in captured_output)

        # second aclgraph with valid output ref, need reconstruct
        y = torch.randn([5, 2])
        with capture_logger() as stdout:
            res2 = model(y, c)
        captured_output = stdout.getvalue()
        self.assertTrue("After reconstructing fx_graph" in captured_output)

        # same graph with mutated_inputs changed, need to rerecapture
        with capture_logger() as stdout:
            res4 = model(x_, c)
        captured_output = stdout.getvalue()
        self.assertTrue("needs to be recaptured" in captured_output)

        # same graph with parameters changed, do not need to rerecapture
        with capture_logger() as stdout:
            model.linear.weight.data = b
            res5 = model(x_, c)
        captured_output = stdout.getvalue()
        self.assertFalse("needs to be recaptured" in captured_output)

        # deleter res5 to make weakref None
        del res5
        with capture_logger() as stdout:
            res1 = model(x, c)
        captured_output = stdout.getvalue()
        self.assertTrue("needs to be recaptured" in captured_output)

    def test_aclgraph_userinput_construct_in_share_memory_with_frozen_parameter(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])
        a = torch.ones(2, 2)
        model.linear.weight.data = a

        model(x)
        b = torch.zeros(2, 2)
        model.linear.weight.data = b

        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_no_frozen_parameter(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        config = CompilerConfig()
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([3, 2])
        a = torch.ones(2, 2)
        model.linear.weight.data = a

        model(x)
        b = torch.zeros(2, 2)
        model.linear.weight.data = b

        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertFalse(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Not Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"found in logs: {cm.output}"
        )

    def test_aclgraph_not_assert_size_stride_empty_tensor(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                return x

        model = Model()
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        aclgraph_backend = torchair.get_npu_backend(compiler_config=config)

        model = torch.compile(model, backend=aclgraph_backend, dynamic=True)
        x = torch.randn([0, 512, 1, 64])
        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertFalse(
            any("assert_size_stride(args[" in log for log in cm.output),
            f"Expect that DEBUG 'assert_size_stride(args['"
            f"not found in logs: {cm.output}"
        )


if __name__ == '__main__':
    unittest.main()
