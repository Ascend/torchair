import os
import sys
import unittest
from unittest.mock import Mock
import logging

import torch
import torch.nn.functional as F
from torch import fx
from torch._functorch.aot_autograd import aot_module_simplified
import sympy

from npugraph_ex.configs.compiler_config import CompilerConfig
from npugraph_ex.core.utils import logger
from npugraph_ex._acl_concrete_graph.utils import reconstruct_args_kwargs, WeakRef, LazyMessage
from npugraph_ex.configs.compiler_config import _process_kwargs_options




from npugraph_ex.configs._option_base import CallableValue
from npugraph_ex._utils.graph_transform_observer import DebugContext
from npugraph_ex import compile_fx

torch._logging.set_logs(dynamo=logging.INFO)
logger.setLevel(logging.DEBUG)

from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
    register_custom_ops,
)
from torchair_st_utils import capture_logger, generate_faked_module, register_is_npu, create_cat_optimization_pass_wrapper

torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())

def get_npugraph_ex_backend():
    def _exec(*args, **kwargs):
        import npugraph_ex       
        config = npugraph_ex.CompilerConfig()
        config.mode = "npugraph_ex"
        return npugraph_ex.get_npu_backend(compiler_config=config)(*args, **kwargs)
    return _exec

def register_npugraph_ex_backend():
    from torch._dynamo.backends.registry import _BACKENDS
    if "npugraph_ex" not in _BACKENDS.keys():
        from torch._dynamo import register_backend as _register_npu_backend
        npugraph_ex_backend = get_npugraph_ex_backend()
        _register_npu_backend(npugraph_ex_backend, "npugraph_ex")

def reset_debug_ctx():
    os.environ['TORCH_COMPILE_DEBUG'] = '0'
    DebugContext.model_cnt = -1
    DebugContext.compile_fx_cnt = -1

class NpugraphExSt(unittest.TestCase):
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

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__
        self.optimize_bak = AclConcreteGraph.optimize_graph_without_runtime
        from npugraph_ex.inference._cache_compiler import CacheBackend
        self.cachebackend_fw_compiler = CacheBackend.fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        self.optimize_cat_with_out_tensor = cat_optimization.optimize_cat_with_out_tensor
        register_npugraph_ex_backend()
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak
        AclConcreteGraph.optimize_graph_without_runtime = self.optimize_bak
        from npugraph_ex.inference._cache_compiler import CacheBackend
        CacheBackend.fw_compiler = self.cachebackend_fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = self.optimize_cat_with_out_tensor
        reset_debug_ctx()
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
        model = torch.compile(Model(), backend="npugraph_ex", options={"clone_input": False}, dynamic=False)
        x = torch.randn([3, 2])
        for i in range(2):
            model(x)
    
    def test_fx_node_shape_analysis(self):
        class Dim:
            def __init__(self, node: fx.Node):
                self.node = node

            def __repr__(self):
                return str(self.node.expr)

        def create_placeholder_node(graph, name, expr) -> fx.Node:
            node = graph.placeholder(name)
            node.expr = expr
            dim = Dim(node)
            return dim

        graph = fx.Graph()
        sym_nodes = {}
        sym_int3 = sympy.Integer(3)
        sym_nodes[sym_int3] = create_placeholder_node(graph, "arg0_1", sym_int3)
        sym_str1 = sympy.symbols('s1')
        sym_nodes[sym_str1] = create_placeholder_node(graph, "arg1_1", sym_str1)
        sym_str2 = sympy.symbols('s2')
        sym_nodes[sym_str2] = create_placeholder_node(graph, "arg2_1", sym_str2)
        sym_float3 = sympy.Float(3.14)
        sym_nodes[sym_float3] = create_placeholder_node(graph, "arg3_1", sym_float3)
        sym_bool1 = sympy.true
        sym_nodes[sym_bool1] = create_placeholder_node(graph, "arg4_1", sym_bool1)

        from npugraph_ex._acl_concrete_graph.acl_graph import construct_fx_node_shape
        ori_shape_list = [1, 2, sym_nodes[sym_int3], sym_nodes[sym_str1], sym_nodes[sym_str2]]
        out_shape = construct_fx_node_shape(ori_shape_list, sym_nodes, 0)
        out_shape = [str(ss) for ss in out_shape]
        target_shape = [1, 2, 3, sym_str1, sym_str2]
        target_shape = [str(ss) for ss in target_shape]
        self.assertTrue(target_shape == out_shape)

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
    def test_aclgraph_capture_and_replay_keep_inference_input_mutations_true(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                x.mul_(2)
                return x + 1
        model = torch.compile(Model(), backend="npugraph_ex",options={"clone_input": False}, dynamic=False)
        x_ = torch.randn([3, 2])
        x = x_.clone()

        # warm up
        model(x_)

        # inference
        with self.assertLogs(logger, level="DEBUG") as cm:
            for _ in range(2):
                output = model(x)

        print(cm.output)
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
        options = {"inplace_pass": True, "clone_input": False}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
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
        from npugraph_ex._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
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
        options = {"clone_input": False, "inplace_pass": True}
        model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
        output, indices = F.max_pool1d(
            torch.randn([1, 1, 4]), 2, stride=2, return_indices=True
        )

        torch._dynamo.mark_static(output)
        torch._dynamo.mark_static(indices)
        model(output, indices, 2)
        model(output, indices, 2)

    def test_aclgraph_custom_update(self):
        from npugraph_ex._acl_concrete_graph.acl_graph import _REPLACE_FUNC_MAP, StaticWorkspaceReplaceFunc
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
        options = {"clone_input": False, "remove_cat_ops": False, "input_inplace_pass": True}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {"clone_input": False, "input_inplace_pass": True,}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
        x = torch.randn([4, 3])
        res = model(x)
        first_model_id = id(model)
        x2 = torch.randn([5, 3])
        res = model(x2)
        second_model_id = id(model)
        self.assertTrue(first_model_id == second_model_id)

    def test_aclgraph_dynamic_sym_in_scale_and_tensor(self):
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
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
        options = {"clone_input": False, "input_inplace_pass": True, "clone_output": True}
        model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
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

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
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
        options = {"clone_input": False, "remove_noop_ops":  False}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
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
        options = {"clone_input": False, "input_inplace_pass": True}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
        x = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x)

        self.assertTrue(
            any("processing reinplace_input_mutated_ops_pass" in log for log in cm.output),
            f"Expected DEBUG log 'processing reinplace_input_mutated_ops_pass' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
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
        options = {"clone_input": False, "input_inplace_pass": True}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
        x = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x)

        self.assertTrue(
            any("processing reinplace_inplaceable_ops_pass" in log for log in cm.output),
            f"Expected DEBUG log 'processing reinplace_inplaceable_ops_pass' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
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
        options = {"clone_input": False, "input_inplace_pass": True, "remove_noop_ops": False}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
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
        options = {"clone_input": False,"input_inplace_pass": True}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
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

    @unittest.skipIf(torch.__version__ < "2.5", "reinplace is unsupported when torch < 2.5")
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
        options = {"clone_input": False, "input_inplace_pass": False, "remove_noop_ops":  False}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
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
        options = {"clone_input": False, "inplace_pass": True}
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)
        x = torch.randn([8, 8])
        y = torch.randn([8, 8])
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model(x, y)

        self.assertTrue(
            any("call_function[target=torch.ops.aten.add_.Tensor]" in log for log in cm.output),
            f"Expected no DEBUG log 'call_function[target=torch.ops.aten.add_.Tensor]' in logs: {cm.output}"
        )

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
        options = {"clone_input": False, "input_inplace_pass": True}

        # when only one graph, no need reconstruct
        model1 = Model()
        model1 = torch.compile(model1, backend="npugraph_ex", options=options, dynamic=True)
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

    # def test_aclgraph_dynamic_disable_mempool_reuse_in_same_fx(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.linear1 = torch.nn.Linear(2, 2)
    #             self.linear2 = torch.nn.Linear(2, 2)
    #
    #         def forward(self, input, bias):
    #             ln1 = self.linear1(input)
    #             ln2 = self.linear2(input)
    #             return ln1, torch.add(ln2, bias)
    #
    #     options = {"clone_input": False, "inplace_pass": True, "reuse_graph_pool_in_same_fx": True,}
    #     model = Model()
    #     model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
    #     x = torch.randn([3, 2])
    #
    #     torch._dynamo.reset()
    #     with capture_logger() as stdout:
    #         model(x, 9.9)
    #     captured_output = stdout.getvalue()
    #     self.assertTrue("memory pool reuse is disable" in captured_output)
    #     self.assertTrue("no mempool reuse in fx_graph" in captured_output)

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

        options = {"clone_input": False, "input_inplace_pass": True}
        from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
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
        model1 = torch.compile(model1, backend="npugraph_ex", options=options, dynamic=True)
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
        model2 = torch.compile(model2, backend="npugraph_ex", options=options, dynamic=True)
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
        options2 = {
            "clone_input": False,
            "input_inplace_pass": True,
            "use_graph_pool": torch.npu.graph_pool_handle()
        }

        model1 = Model1()
        model1 = torch.compile(model1, backend="npugraph_ex", options=options2, dynamic=True)
        with capture_logger() as stdout:
            res = model1(x)
        captured_output = stdout.getvalue()
        self.assertTrue("no mempool reuse in fx_graph" in captured_output)
        pool_id1 = _get_pool_id

        model2 = Model2()
        model2 = torch.compile(model2, backend="npugraph_ex", options=options2, dynamic=True)
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

    # def test_compile_static_kernel(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.linear1 = torch.nn.Linear(2, 2)
    #             self.linear2 = torch.nn.Linear(2, 2)

    #         def forward(self, x):
    #             ln1 = self.linear1(x)
    #             ln2 = self.linear2(x)
    #             return ln1 + ln2

    #     model = Model()
    #     options = {"clone_input": False, "input_inplace_pass": True, "static_kernel_compile": True,}
    #     model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=False)
    #     x = torch.randn([3, 2])
    #     from npugraph_ex.core import _torchair
    #     _torchair.GetSocName()
    #     _torchair.AclopStartDumpArgs(1, "..")
    #     _torchair.AclopStopDumpArgs(1)

    #     import warnings
    #     with warnings.catch_warnings(record=True) as caught:
    #         warnings.simplefilter("always")
    #         try:
    #             model(x)
    #         except ValueError as e:
    #             messages = [str(w.message) for w in caught]
    #             self.assertTrue(
    #                 any("The current version now supports caching run packages from static kernel compilation" in m for m in messages),
    #                 f"Expected warning 'The current version now supports caching run packages from static kernel compilation' not found in {messages}"
    #             )


    def test_aclgraph_supported_blocking_env(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + x

        options = {"clone_input": False, "input_inplace_pass": True}
        model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=False)

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

        options = {"clone_input": False, "input_inplace_pass": True}

        # test use default static_capture_size_limit, and do not fall back to eager
        torch._dynamo.reset()
        model1 = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
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
        options = options = {"clone_input": False, "input_inplace_pass": True, "capture_limit": 1}
        torch._dynamo.reset()
        model1 = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([4, 2]))
        self.assertTrue("capture_limit reached" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        # fall back to eager no aclgraph log
        self.assertTrue("Find captured AclGraph" not in stdout.getvalue())
        self.assertTrue("Success to capture fx_graph" not in stdout.getvalue())

    def test_aclgraph_recapture_non_mutated_input_with_address_change_and_input_with_no_address_change(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + 1

        model = Model()
        options = {"input_inplace_pass": True,"clone_input": False}

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {"clone_input": False, "input_inplace_pass": True}

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {"input_inplace_pass": True, "clone_input": False}

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {"input_inplace_pass": True, "clone_input": False}

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {"frozen_parameter": True, "input_inplace_pass": True, "clone_input": False}

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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

    # @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    # def test_multi_mutated_input(self):
    #     def f(x, out_sin, out_cos):
    #         return torch.ops.custom.sin_cos_inplace.default(x, out_sin, out_cos)

    #     options = {"clone_input": False}
    #     x = torch.randn(3)
    #     sin = torch.randn(3)
    #     cos = torch.randn(3)
    #     model = torch.compile(f, backend="npugraph_ex", options=options)
    #     with self.assertLogs(logger, level="DEBUG") as cm:
    #         res = model(x, sin, cos)
    #     self.assertTrue(
    #         any("call_function[target=torch.ops.custom.sin_cos_inplace.default]" in log for log in cm.output),
    #         f"Expected DEBUG log 'call_function[target=torch.ops.custom.sin_cos_inplace.default]' in logs: {cm.output}"
    #     )

    # @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    # def test_multi_mutated_input_with_view_before(self):
    #     def f(x, out_sin, out_cos):
    #         sin_view = out_sin.view(-1, 1)
    #         y = torch.ops.custom.sin_cos_inplace.default(x, sin_view, out_cos)
    #         res = out_cos + 1
    #         return y, res

    #     options = {"clone_input": False}
    #     x = torch.ones(3)
    #     sin = torch.ones(3)
    #     cos = torch.ones(3)
    #     model = torch.compile(f, backend="npugraph_ex", options=options)
    #     with self.assertLogs(logger, level="DEBUG") as cm:
    #         res = model(x, sin, cos)
    #     self.assertTrue(
    #         any("reinplace failed" in log for log in cm.output),
    #         f"Expected DEBUG log 'reinplace failed' in logs: {cm.output}"
    #     )

    # @unittest.skipIf(torch.__version__ < "2.2", "torch._functionalize_replace is unsupported when torch < 2.2")
    # def test_multi_mutated_input_with_view_after(self):
    #     def f(x, out_sin, out_cos):
    #         y = torch.ops.custom.sin_cos_inplace.default(x, out_sin, out_cos)
    #         sin_view = out_sin.view(-1, 1)
    #         res = sin_view + 1
    #         return y, res

    #     options = {"clone_input": False}
    #     x = torch.ones(3)
    #     sin = torch.ones(3)
    #     cos = torch.ones(3)
    #     model = torch.compile(f, backend="npugraph_ex", options=options)
    #     with self.assertLogs(logger, level="DEBUG") as cm:
    #         res = model(x, sin, cos)
    #     self.assertTrue(
    #         any("call_function[target=torch.ops.custom.sin_cos_inplace.default]" in log for log in cm.output),
    #         f"Expected DEBUG log 'call_function[target=torch.ops.custom.sin_cos_inplace.default]' in logs: {cm.output}"
    #     )

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

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        call_bak = AclConcreteGraph.__call__
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        try:
            options ={"inplace_pass": True}
            compiled_model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
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

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        options ={"clone_input": False}
        compiled_model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
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

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = wrapper_call(AclConcreteGraph.__call__)

        options = {"clone_input": False, "remove_noop_ops": False}
        compiled_model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=True)
        compiled_model(x=torch.randn([2, 2]))


    # Generate patterns with auto-incremented indics
    def check_debug_dump_full_files(self, root_path, sub_dir="model", phases=["forward", "backward"], high_version=False):

        # Define templates for ACL mode (without hardcoded indices)
        ACL_COMMON_FILES = [
            *(["dynamo_out_graph.txt"] if phases!=["compile_fx"] else [])
        ]

        ACL_STEP_TEMPLATS = [
            "aot_{phase}_graph.txt",
            "aot_{phase}_graph_after_post_grad_custom_pre_pass.txt",
            "aot_{phase}_graph_after_optimize_noop_ops.txt",
            "aot_{phase}_graph_after_recover_view_inplace_pattern.txt",
            "aot_{phase}_graph_after_apply_pattern_passes.txt",
            "aot_{phase}_graph_after_view_to_reshape.txt",
            "aot_{phase}_graph_after_post_grad_custom_post_pass.txt",
            "aot_{phase}_graph_after_remove_cat_ops.txt",
            "aot_{phase}_graph_after_apply_event_closure_with_multi_stream.txt",
            "aot_{phase}_graph_after_apply_event_record.txt",
            "aot_{phase}_graph_after_eliminate_dead_code.txt",
            *(["aot_{phase}_graph_after_reinplace_inplaceable_ops_pass.txt"] if high_version else []),
            "aot_{phase}_graph_after_reinplace_input_mutated_ops.txt",
            *(["aot_{phase}_graph_after_decompose_auto_functionalized.txt"] if high_version else []),
            "aot_{phase}_graph_after_eliminate_self_copy.txt",
            "aot_{phase}_graph_after_replace_dynamic_workspace_ops.txt",
            "aot_{phase}_graph_after_replace_core_limit_nodes.txt",
            "aot_{phase}_graph_after_resolve_default_stream_markers.txt",
        ]

        patterns = []
        # Add common files
        for file in ACL_COMMON_FILES:
            patterns.append(f"{sub_dir}__{{id}}/{file}")
        for phase in phases:
            for idx, template in enumerate(ACL_STEP_TEMPLATS):
                formatted_name = template.format(phase=phase)
                patterns.append(f"{sub_dir}__{{id}}/{phase}/{idx:03d}_{formatted_name}")
            patterns.append(f"{sub_dir}__{{id}}/{phase}/output_code.py")

        EXPECTED_FILE_PATTERNS_ACL = patterns
        # 2. Verify all expected files exist
        expected_files = []
        for template in EXPECTED_FILE_PATTERNS_ACL:
            rel_path = template.format(id=0)
            expected_files.append(rel_path)

        # Collect actual files first for error reporting
        actual_files = []
        for root, _, files in os.walk(root_path):
            for f in files:
                actual_files.append(os.path.join(root, f))
        actual_msg = "Actual files:\n" + "\n".join(f"  - {f}" for f in actual_files) if actual_files else "Actual files: (empty)"

        def check_torchair_directory_structure(base_dir: str, file_list: list) -> list:
            missing_files = []
            for rel_path in file_list:
                abs_path = os.path.join(base_dir, rel_path)
                if not os.path.exists(abs_path):
                    missing_files.append(abs_path)
            return missing_files
        missing_files = check_torchair_directory_structure(root_path, expected_files)
        if missing_files:
            missing_msg = "Missing files:\n" + "\n".join(f"  - {f}" for f in missing_files) + "\n\n" + actual_msg
            self.assertFalse(missing_files, msg=missing_msg)

        # Check file count to ensure no extra files
        expected_count = len(EXPECTED_FILE_PATTERNS_ACL)
        actual_count = len(actual_files)
        if actual_count != expected_count:
            self.assertEqual(
                actual_count,
                expected_count,
                msg=f"File count mismatch: expected {expected_count} files, got {actual_count} files\n{actual_msg}"
            )
        return patterns
        
    @unittest.skipIf(torch.__version__ < "2.6", "")
    def test_torch_compile_acl_debug_dump(self):

        from torch._dynamo.utils import get_debug_dir
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchair_debug_") as tmpdir, \
            torch._dynamo.config.patch(debug_dir_root=tmpdir):
            self.assertIsNotNone(tmpdir)
            os.environ['TORCH_COMPILE_DEBUG'] = '1'

            def _custom_pre_fn(gm, example_inputs, config: CompilerConfig):
                return None

            def _custom_post_fn(gm, example_inputs, config: CompilerConfig):
                return None
            
            options = {"remove_noop_ops":  True, "post_grad_custom_pre_pass": _custom_pre_fn, "post_grad_custom_post_pass": _custom_post_fn}
            class Model(torch.nn.Module):
                def forward(self, x):
                    return 2 * x

            model = Model()
            compiled_model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
            x = torch.randn(10, 10, requires_grad=True)
            out = compiled_model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.randn(10, 10)
            loss = loss_fn(out, target)
            loss.backward()

            debug_dir_output = get_debug_dir()
            # 1. Verify the existence of the torchair directory
            torchair_root = os.path.join(debug_dir_output, "npugraph_ex")
            self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

            self.check_debug_dump_full_files(torchair_root, high_version=True)


    @unittest.skipIf(torch.__version__ > "2.2", "")
    def test_torch_compile_acl_debug_dump_low_version(self):

        from torch._dynamo.utils import get_debug_dir
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchair_debug_") as tmpdir, \
            torch._dynamo.config.patch(debug_dir_root=tmpdir):
            self.assertIsNotNone(tmpdir)
            os.environ['TORCH_COMPILE_DEBUG'] = '1'

            def _custom_pre_fn(gm, example_inputs, config: CompilerConfig):
                return None

            def _custom_post_fn(gm, example_inputs, config: CompilerConfig):
                return None

            options = {
                "remove_noop_ops":  True,
                "clone_input": False,
                "post_grad_custom_pre_pass": _custom_pre_fn,
                "post_grad_custom_post_pass": _custom_post_fn,
                "input_inplace_pass": True
            }

            class Model(torch.nn.Module):
                def forward(self, x):
                    return 2 * x

            model = Model()
            compiled_model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
            x = torch.randn(10, 10, requires_grad=True)
            out = compiled_model(x)
            loss_fn = torch.nn.MSELoss()
            target = torch.randn(10, 10)
            loss = loss_fn(out, target)
            loss.backward()

            debug_dir_output = get_debug_dir()
            # 1. Verify the existence of the torchair directory
            torchair_root = os.path.join(debug_dir_output, "npugraph_ex")
            self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")
            self.check_debug_dump_full_files(torchair_root, high_version=False)

    @unittest.skipIf(torch.__version__ < "2.6", "")
    def test_compile_fx_debug_dump(self):
        
        from torch._dynamo.utils import get_debug_dir
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchair_debug_") as tmpdir, \
            torch._dynamo.config.patch(debug_dir_root=tmpdir):
            self.assertIsNotNone(tmpdir)
            os.environ['TORCH_COMPILE_DEBUG'] = '1'

            def _custom_pre_fn(gm, example_inputs, config: CompilerConfig):
                return None

            def _custom_post_fn(gm, example_inputs, config: CompilerConfig):
                return None
            
            options = {"remove_noop_ops":  True, "post_grad_custom_pre_pass": _custom_pre_fn, "post_grad_custom_post_pass": _custom_post_fn}
            class Model(torch.nn.Module):
                def forward(self, x):
                    return 2 * x

            def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
              
                compiled_graph = compile_fx(
                    gm,
                    example_inputs,
                    options
                )
                return compiled_graph

            def custom_backend(gm: torch.fx.GraphModule, example_inputs):
                return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)

            model = Model()
            compiled_model = torch.compile(model, backend=custom_backend, dynamic=True)
            x = torch.randn(10, 10)
            out = compiled_model(x)

            debug_dir_output = get_debug_dir()
            # 1. Verify the existence of the torchair directory
            torchair_root = os.path.join(debug_dir_output, "npugraph_ex")
            self.assertTrue(os.path.exists(torchair_root), msg=f"torchair directory does not exist: {torchair_root}")

            self.check_debug_dump_full_files(torchair_root, sub_dir="compile", phases=["compile_fx"], high_version=True)

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
        options = {
            "clone_input": False,
            "input_inplace_pass": True
        }

        # when only one graph, no need reconstruct
        model = Model()
        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {
            "frozen_parameter": True,
            "input_inplace_pass": True,
            "clone_input": False,
        }

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {
            "input_inplace_pass": True,
            "clone_input": False
        }

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
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
        options = {
            "clone_input": False,
            "input_inplace_pass": True
        }

        model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=True)
        x = torch.randn([0, 512, 1, 64])
        with self.assertLogs(logger, level="DEBUG") as cm:
            model(x)
        self.assertFalse(
            any("assert_size_stride(args[" in log for log in cm.output),
            f"Expect that DEBUG 'assert_size_stride(args['"
            f"not found in logs: {cm.output}"
        )

    def _make_optimize_wrapper(self):
        """Returns a wrapped optimize_graph_without_runtime that captures the optimized fx_graph."""
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        original_func = AclConcreteGraph.optimize_graph_without_runtime
        captured = []

        def wrapper(self_graph, *sample_args, observer=None, aot_gm=None):
            ret = original_func(self_graph, *sample_args, observer=observer, aot_gm=aot_gm)
            captured.append(self_graph.fx_graph)
            return ret

        AclConcreteGraph.optimize_graph_without_runtime = wrapper
        return original_func, captured

    def _assert_no_self_copy_and_check_precision(self, model_class, options, *inputs):
        """Compile model with graph inspection, assert no copy_(x,x), compare with eager."""
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        original_func, captured_fx_graphs = self._make_optimize_wrapper()
        compiled = torch.compile(model_class(), backend="npugraph_ex", options=options)
        result = compiled(*inputs)
        for fx_gm in captured_fx_graphs:
            for node in fx_gm.graph.nodes:
                num_users = len(node.users)
                if node.target is torch.ops.aten.copy_.default and len(node.args) >= 2 and num_users == 0:
                    self.assertNotEqual(
                        id(node.args[0]), id(node.args[1]),
                        f"self-copy copy_({node.args[0].name}, {node.args[1].name}) "
                        f"should have been eliminated by eliminate_self_copy pass"
                    )
        return result

    def test_self_copy_no_crash(self):
        """copy_(x, x) should be eliminated, result precision matches eager."""
        class Model1(torch.nn.Module):
            def forward(self, x):
                x.copy_(x)
                return

        x1 = torch.randn([3])
        x2 = x1.clone()
        Model1()(x1)
        self._assert_no_self_copy_and_check_precision(
            Model1, {"clone_input": False, "input_inplace_pass": True}, x2)
        self.assertTrue(torch.equal(x1, x2))

        class Model2(torch.nn.Module):
            def forward(self, x):
                x.copy_(x)
                return x

        x = torch.randn([3])
        eager = Model2()(x.clone())
        result = self._assert_no_self_copy_and_check_precision(
            Model2, {"clone_input": False, "input_inplace_pass": True}, x)
        self.assertTrue(torch.equal(eager, result))

    def test_self_copy_user_write_chain(self):
        """Chained copy_(x,x) -> copy_(y,y) should all be eliminated."""
        class Model(torch.nn.Module):
            def forward(self, x):
                y = x.copy_(x)
                z = y.copy_(y)
                return

        x1 = torch.randn([3])
        x2 = x1.clone()
        Model()(x1)
        result = self._assert_no_self_copy_and_check_precision(
            Model, {"clone_input": False, "input_inplace_pass": True}, x2)
        self.assertTrue(torch.equal(x1, x2))

    def test_self_copy_preserves_normal_copy(self):
        """copy_(y, x) where x != y must NOT be eliminated."""
        class Model(torch.nn.Module):
            def forward(self, x):
                y = torch.zeros_like(x)
                y.copy_(x)
                # copy = torch.ops.aten.copy_.default(zeros_like, arg0_1)
                # return zeros_like
                return

        x1 = torch.randn([3])
        x2 = x1.clone()
        Model()(x1)
        self._assert_no_self_copy_and_check_precision(
            Model, {"clone_input": False, "input_inplace_pass": True}, x2)
        self.assertTrue(torch.equal(x1, x2))

    def test_self_copy_with_arithmetic(self):
        """Self-copy followed by arithmetic — precision must match eager."""
        class Model(torch.nn.Module):
            def forward(self, x):
                x.copy_(x)
                y = x + 1
                return y

        x = torch.randn([3])
        eager = Model()(x.clone())
        result = self._assert_no_self_copy_and_check_precision(
            Model, {"clone_input": False, "input_inplace_pass": True}, x)
        self.assertTrue(torch.equal(eager, result))

    def test_self_copy_return_different_alias(self):
        """Self-copy on input, then x - 1 returned — precision must match eager."""
        class Model(torch.nn.Module):
            def forward(self, x):
                x.copy_(x)
                y = x - 1
                return y

        x = torch.randn([3])
        eager = Model()(x.clone())
        result = self._assert_no_self_copy_and_check_precision(
            Model, {"clone_input": False, "input_inplace_pass": True}, x)
        self.assertTrue(torch.equal(eager, result))

    def test_self_copy_with_decompose(self):
        """Custom mutable op triggers auto_functionalized -> decompose -> eliminate_self_copy."""

        lib = torch.library.Library("test_decompose", "FRAGMENT")
        lib.define("sin_inplace(Tensor x, Tensor(a!) result) -> None")

        @torch.library.impl(lib, "sin_inplace", "Meta")
        def sin_inplace_meta(x, result):
            pass

        @torch.library.impl(lib, "sin_inplace", "CPU")
        def sin_inplace(x, result):
            result.copy_(x.sin())

        class Model(torch.nn.Module):
            def forward(self, x):
                y = x
                torch.ops.test_decompose.sin_inplace(torch.zeros_like(x), y)
                return

        x1 = torch.randn([3])
        x2 = x1.clone()
        Model()(x1)
        self._assert_no_self_copy_and_check_precision(
            Model, {"clone_input": False, "input_inplace_pass": True}, x2)
        self.assertTrue(torch.equal(x1, x2))

    # def test_aclgraph_core_limit_with_static_kernel(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()

    #         def forward(self, x):
    #             with npugraph_ex.scope.limit_core_num(12, 24):
    #                 output = torch.square(x)
    #             return output

    #     options = {
    #         "static_kernel_compile": True
    #     }
    #     model = torch.compile(Model(), backend="npugraph_ex", options=options, dynamic=False)
    #     x = torch.randn([5, 5])

    #     import warnings
    #     with warnings.catch_warnings(record=True) as caught:
    #         warnings.simplefilter("always")
    #         try:
    #             model(x)
    #         except Exception:
    #             pass
    #         target_warning = "When both static shape kernel and core limit are enabled"
    #         messages = [str(w.message) for w in caught]
    #         self.assertFalse(
    #             any(target_warning in m for m in messages),
    #             f"Expected warning '{target_warning}' found in {messages}"
    #         )
    #         self.assertTrue(config.experimental_config.aclgraph._aclnn_static_shape_kernel)

    def test_capture_and_recapture_cnt(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            @torch.no_grad()
            def forward(self, x, y):
                z = self.linear(x)
                y.add_(1)
                return z

        from torch._dynamo.utils import counters
        options = {"capture_limit": 3}

        model = Model()
        opt_model = torch.compile(model, backend="npugraph_ex", options=options, dynamic=False)

        # capture graph1
        x1 = torch.randn([10, 32])
        y1 = torch.randn([10, 32])
        with self.assertLogs(logger, level="INFO") as cm:
            opt_model(x1, y1)
        from npugraph_ex.npu_fx_compiler import _GLOBAL_GRAPH_ID as graph_1_id
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertFalse(
            any(f"For graph graph_{graph_1_id} : the count of captures is" in log for log in cm.output),
            f"Not Expect that DEBUG 'For graph graph_{graph_1_id} : the count of captures is'"
            f"found in logs: {cm.output}"
        )
        self.assertFalse(
            any("[npugraph_ex overhead]" in log for log in cm.output),
            f"Not Expect that DEBUG '[npugraph_ex overhead]'"
            f"found in logs: {cm.output}"
        )

        # dtype changed, capture graph2 with graph_key1
        x2 = torch.randn([20, 32])
        y2 = torch.randn([20, 32])
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x2, y2)
        from npugraph_ex.npu_fx_compiler import _GLOBAL_GRAPH_ID as graph_2_id
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_2_id}"], 1)

        # weigth data change, capture graph2 with graph_key2
        b = torch.zeros([32,32])
        x3 = torch.randn([20, 32])
        y3 = torch.randn([20, 32])
        opt_model.linear.weight.data = b
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x3, y3)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_2_id}"], 2)

        # mutated input change, recapture graph2 with graph_key2
        x4 = torch.randn([20, 32])
        y4 = torch.randn([20, 32])
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x4, y4)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_2_id}"], 2)
        self.assertEqual(counters["npugraph_ex"][f"recapture_due_to_mutated_input_change_graph_{graph_2_id}"], 1)

        # mutated input change, recapture graph2 with graph_key2 again
        x5 = torch.randn([20, 32])
        y5 = torch.randn([20, 32])
        with self.assertLogs(logger, level="DEBUG") as cm5:
            opt_model(x5, y5)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_2_id}"], 2)
        self.assertEqual(counters["npugraph_ex"][f"recapture_due_to_mutated_input_change_graph_{graph_2_id}"], 2)
        self.assertTrue(
            any(f"For graph graph_{graph_2_id} : the count of captures is 2" in log for log in cm5.output),
            f"Expect that DEBUG 'For graph graph_2 : the count of captures is 2'"
            f"not found in logs: {cm5.output}"
        )
        self.assertTrue(
            any("and count of recaptures caused by mutated inputs changed is 2" in log for log in cm5.output),
            f"Expect that DEBUG 'and count of recaptures caused by mutated inputs changed is 2'"
            f"not found in logs: {cm5.output}"
        )
        expected_sequence = [
            "[npugraph_ex overhead] generate graph_key",
            "[npugraph_ex overhead] process input",
            "[npugraph_ex overhead] get updated params"
        ]
        last_pos = -1
        for phrase in expected_sequence:
            found = False
            for i in range(last_pos + 1, len(cm5.output)):
                if phrase in cm.output[i]:
                    last_pos = i
                    found = True
                    break
            self.assertTrue(
                found,
                f"Expected log phrase '{phrase}' not found in correct order. Full logs: {cm5.output}"
            )

        # weigth data change and mutated change, capture graph2 with graph_key3 and recapture, then reach static_capture_size_limit(3)
        b1 = torch.zeros([32, 32])
        x6 = torch.randn([20, 32])
        y6 = torch.randn([20, 32])
        y7 = torch.randn([20, 32])
        opt_model.linear.weight.data = b1
        with self.assertLogs(logger, level="DEBUG") as cm:
            opt_model(x6, y6)
            opt_model(x6, y7)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_1_id}"], 1)
        self.assertEqual(counters["npugraph_ex"][f"captured_graph_{graph_2_id}"], 0)
        self.assertEqual(counters["npugraph_ex"][f"recapture_due_to_mutated_input_change_graph_{graph_2_id}"], 0)

    def test_debug_compare_fx_graphs(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x, y, z):
                sqrt_01 = torch.sqrt(x)
                softmax_01 = torch.softmax(sqrt_01, dim=-1)
                abs_01 = torch.abs(softmax_01)
                split_01, split_02 = torch.split(abs_01, split_size_or_sections=[6, 6], dim=0)
                matmul_01 = torch.matmul(split_01, y)
                add_01 = torch.add(split_02, matmul_01)
                concat_01 = torch.cat([add_01, z], dim=0)
                relu_01 = self.relu(concat_01)
                transpose_01 = torch.transpose(relu_01, 0, 1)
                return transpose_01

        def parallel_abs_sub_1(gm, example_inputs, config: CompilerConfig):
            fx_graph = gm.graph
            for node in fx_graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.aten.sqrt.default:
                    with fx_graph.inserting_before(node):
                        fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                            ["_user_stream_label"], ["stream0"]))

                if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                    with fx_graph.inserting_after(node):
                        fx_graph.call_function(
                            torch.ops.air.scope_exit.default, args=())

        def parallel_abs_sub_2(gm, example_inputs, config: CompilerConfig):
            fx_graph = gm.graph
            for node in fx_graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.aten._softmax.default:
                    with fx_graph.inserting_before(node):
                        fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                            ["_user_stream_label"], ["stream1"]))

                if node.op == "call_function" and node.target == torch.ops.aten.split_with_sizes.default:
                    with fx_graph.inserting_after(node):
                        fx_graph.call_function(torch.ops.air.scope_exit.default, args=())

        options = {
            "clone_input": False,
            "post_grad_custom_pre_pass": parallel_abs_sub_1,
            "post_grad_custom_post_pass": parallel_abs_sub_2,
        }

        input0 = torch.randn(12, 6, dtype=torch.float32)
        input1 = torch.randn(6, 6, dtype=torch.float32)
        input2 = torch.randn(12, 6, dtype=torch.float32)

        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend="npugraph_ex", options=options, dynamic=False)
        with self.assertLogs(logger, level="DEBUG") as cm:
            npu_mode(input0, input1, input2)

        expected_phrases = [
            "After fx graph pass(view_to_reshape) optimization",
            "After fx graph pass(post_grad_custom_pre_pass) optimization",
            "After fx graph pass(recover_view_inplace_pattern) optimization",
            "After fx graph pass(post_grad_custom_post_pass) optimization",
            "After fx graph pass(optimize_noop_ops) optimization",
            "After fx graph pass(apply_pattern_passes) optimization",
            "After fx graph pass(replace_dynamic_workspace_ops) optimization",
            "After fx graph pass(decompose_auto_functionalized) optimization",
            "After fx graph pass(reinplace_inplaceable_ops_pass) optimization",
            "After fx graph pass(reinplace_input_mutated_ops) optimization",
            "After fx graph pass(remove_cat_ops) optimization",
            "After fx graph pass(apply_event_closure_with_multi_stream) optimization",
            "After fx graph pass(apply_event_record) optimization",
            "After fx graph pass(replace_core_limit_nodes) optimization",
            "After fx graph pass(eliminate_dead_code) optimization",
            "After fx graph pass(all npugraph_ex pass) optimization"
        ]

        for phrase in expected_phrases:
            self.assertTrue(any(phrase in log for log in cm.output),
                            f"Expect that DEBUG '{phrase}' not found in logs: {cm.output}"
                            )

        self.assertTrue(any("before 15 nodes, after 26 nodes" in log for log in cm.output),
                        f"Expect that DEBUG 'before 15 nodes, after 26 nodes' not found in logs: {cm.output}"
                        )

        torch._dynamo.reset()
        with self.assertLogs(logger, level="INFO") as cm:
            npu_mode(input0, input1, input2)

        self.assertFalse(any("After fx graph pass(" in log for log in cm.output),
                        f"Not Expect that DEBUG 'After fx graph pass(' found in logs: {cm.output}"
                        )

    def assert_cat_optimization_success(self, graph_before, graph_after):
        """Verify that cat was replaced with empty + slice + out operations."""
        cat_found_after = False
        empty_found_after = False
        slice_found_after = False
        out_ops_found_after = False
        
        # Check graph after optimization
        for node in graph_after.graph.nodes:
            if node.op == "call_function":
                if node.target == torch.ops.aten.cat.default:
                    cat_found_after = True
                if node.target == torch.ops.aten.empty.memory_format:
                    empty_found_after = True
                if node.target == torch.ops.aten.slice.Tensor:
                    slice_found_after = True
                # Check for operations with 'out' in kwargs
                if node.kwargs and 'out' in node.kwargs:
                    out_ops_found_after = True
        
        # Cat optimization should succeed: cat removed, empty+slice+out ops added
        self.assertFalse(
            cat_found_after,
            "cat node should be removed after optimization"
        )
        self.assertTrue(
            empty_found_after,
            "empty.memory_format node should be created after optimization"
        )
        self.assertTrue(
            slice_found_after,
            "slice.Tensor node should be created after optimization"
        )
        self.assertTrue(
            out_ops_found_after,
            "operations with .out parameter should be created after optimization"
        )

    def assert_optimization_skipped(self, graph_before, graph_after):
        """Verify that optimization was skipped."""
        cat_nodes_before = [n for n in graph_before.graph.nodes 
                            if n.op == "call_function" and n.target == torch.ops.aten.cat.default]
        cat_nodes_after = [n for n in graph_after.graph.nodes 
                            if n.op == "call_function" and n.target == torch.ops.aten.cat.default]
        
        # Cat node should still exist (optimization skipped)
        self.assertEqual(len(cat_nodes_before), len(cat_nodes_after),
                           "Cat node should still exist when optimization is skipped")

    def test_cat_optimization_basic(self):
        """Test basic cat optimization: cat replaced with empty + slice + out operations."""
        def f(x):
            output1 = x.exp()
            output2 = x.exp()
            result = torch.cat([output1, output2], dim=0)
            return result

        x = torch.randn(8, dtype=torch.float32)
        
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = create_cat_optimization_pass_wrapper(self.assert_cat_optimization_success)
        
        options = {}

        model = torch.compile(f, backend="npugraph_ex", options=options, dynamic=True)
        x = torch.randn(8, dtype=torch.float32)
        result = model(x)
        
        expected = torch.cat([x.exp(), x.exp()], dim=0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_cat_optimization_wrong_order_success(self):
        """Test cat optimization with wrong order: now succeeds (no order check)."""
        def f(x):
            output2 = x.exp()
            output1 = x.exp()
            result = torch.cat([output1, output2], dim=0)
            return result

        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = create_cat_optimization_pass_wrapper(self.assert_cat_optimization_success)

        options = {}
        model = torch.compile(f, backend="npugraph_ex", options=options, dynamic=True)
        x = torch.randn(8, dtype=torch.float32)
        result = model(x)

        expected = torch.cat([x.exp(), x.exp()], dim=0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_cat_optimization_skip_non_first_dim(self):
        """Test that cat optimization is skipped for non-first dimension concatenation."""
        def f(x):
            output1 = x.exp()
            output2 = x.exp()
            result = torch.cat([output1, output2], dim=1)
            return result

        x = torch.randn(2, 8, dtype=torch.float32)
        
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = create_cat_optimization_pass_wrapper(self.assert_optimization_skipped)
        
        options = {}
        model = torch.compile(f, backend="npugraph_ex", options=options, dynamic=True)
        result = model(x)

    def test_cat_optimization_skip_no_out_variant(self):
        """Test that cat optimization is skipped when upstream ops don't support .out variant."""
        def f(x):
            # view doesn't support .out variant
            output1 = x.view(-1)
            output2 = x.view(-1)
            result = torch.cat([output1, output2], dim=0)
            return result

        x = torch.randn(8, dtype=torch.float32)
        
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = create_cat_optimization_pass_wrapper(self.assert_optimization_skipped)
        
        options = {}
        model = torch.compile(f, backend="npugraph_ex", options=options, dynamic=True)
        result = model(x)

    def test_cat_optimization_different_ops(self):
        """Test cat optimization with different upstream operations."""
        def f(x, y):
            output1 = x.exp()
            output2 = x.sin()
            output3 = x.clone()
            output3.add_(y)
            result = torch.cat([output1, output2, output3], dim=0)
            return result
        
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = create_cat_optimization_pass_wrapper(self.assert_cat_optimization_success)
        
        options = {}
        model = torch.compile(f, backend="npugraph_ex", options=options, dynamic=True)
        x = torch.randn(8, 3, dtype=torch.float32)
        y = torch.randn(8, 3, dtype=torch.float32)
        result = model(x, y)

        expected = torch.cat([x.exp(), x.sin(), x + y], dim=0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))
        

    def test_npugraph_ex_process_kwargs_options_invalid_option(self):
        config = CompilerConfig()

        test_kwargs = {
            "options": {
                "inplace_pass": 'test',
                "input_inplace_pass": 'test',
                "reuse_graph_pool_in_same_fx": 'test'
            }
        }

        try:
            _process_kwargs_options(config, test_kwargs)
        except Exception as e:
            assert str(e).__contains__(
                "(type: <class 'str'>) not in optional list [True, False] (type: <class 'bool'>)")


    def test_compile_fx(self):
        """Test compile_fx with torch.compile and custom_backend."""
        
        from npugraph_ex import npu_fx_compiler

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output
        
        captured_config = None
        original_get_compiler = npu_fx_compiler.get_compiler

        def wrapped_get_compiler(config):
            nonlocal captured_config
            captured_config = config
            return original_get_compiler(config)

        # 使用 compile_fx 编译图
        test_options = {
                "static_kernel_compile": True,
                "inplace_pass": False,
                "input_inplace_pass": False,
                "remove_noop_ops": False,
                "remove_cat_ops": False,
                "force_eager": False,
                "pattern_fusion_pass": False,
                "clone_input": False,
                "frozen_parameter": False,
                "reuse_graph_pool_in_same_fx": False,
                "capture_limit": 64,
                "clone_output": False,
                "dump_tensor_data": False,
                "data_dump_stage": "optimized",
                "data_dump_dir": "./"
            }

        def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
            npu_fx_compiler.get_compiler = wrapped_get_compiler
            try:
                compiled_graph = compile_fx(
                    gm,
                    example_inputs,
                    test_options
                )
            finally:
                npu_fx_compiler.get_compiler = original_get_compiler
            return compiled_graph

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)
        
        # 准备数据
        x1 = torch.randn(4, 64, 512, dtype=torch.float16)
        x2 = torch.randn(64, 512, 128, dtype=torch.float16)
        model = DsModel()
        # 编译并执行
        model = torch.compile(model, backend=custom_backend, dynamic=False, fullgraph=True)
        result = model(x1, x2)
        eager_output = model(x1, x2)
        self.assertTrue(torch.allclose(result, eager_output))

        assert captured_config.static_kernel_compile.value
        assert captured_config.inplace_pass.value is False
        assert captured_config.input_inplace_pass.value is False
        assert captured_config.remove_noop_ops.value is False
        assert captured_config.remove_cat_ops.value is False
        assert captured_config.force_eager.value is False
        assert captured_config.pattern_fusion_pass.value is False
        assert captured_config.clone_input.value is False
        assert captured_config.frozen_parameter.value is False
        assert captured_config.reuse_graph_pool_in_same_fx.value is False
        assert captured_config.capture_limit.value == "64"
        assert captured_config.clone_output.value is False
        assert captured_config.dump_tensor_data.value is False
        assert captured_config.data_dump_stage.value == 'optimized'
        assert captured_config.data_dump_dir.value == './'

    def test_capture_error_mode_option(self):
        def f(x):
            return x + 1
        
        input = torch.randn([2, 2])

        def test_mode(mode = None):
            options = {}
            if mode is not None:
                options = {"capture_error_mode": mode}
            compile_f = torch.compile(f, backend="npugraph_ex", options=options)
            with self.assertLogs(logger, level="DEBUG") as cm:
                compile_f(input)
            if mode is None:
                mode = "global"
            match_log = f"capture_error_mode=\"{mode}\""
            return any(match_log in log for log in cm.output)
        
        # 测试capture_error_mode不配置和分别为global, thread_local, relaxed
        self.assertTrue(test_mode())
        self.assertTrue(test_mode("global"))
        self.assertTrue(test_mode("thread_local"))
        self.assertTrue(test_mode("relaxed"))

        # capture_error_mode配置为global, thread_local, relaxed以外的选项报错
        options = {"capture_error_mode": "xxxxx"}
        compile_f = torch.compile(f, backend="npugraph_ex", options=options)
        try:
            compile_f(input)
        except Exception as e:
            assert str(e).__contains__("not in optional list ['global', 'thread_local', 'relaxed']")

if __name__ == '__main__':
    unittest.main()
