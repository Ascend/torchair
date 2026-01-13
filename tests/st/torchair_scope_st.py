import os
import time
import logging
import json
from pathlib import Path

import torch
import torchair

from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
from torchair.configs.compiler_config import CompilerConfig

import unittest

os.environ['TNG_LOG_LEVEL'] = '0'
logger.setLevel(logging.DEBUG)


class TorchairSt(unittest.TestCase):
    def setUp(self) -> None:
        self.call_bak = GeConcreteGraph.__call__
        self.optimize_bak = GeConcreteGraph.optimize_graph_without_runtime
        torchair.core._backend._GLOBAL_COMPILE_OPTION = None
        return super().setUp()


    def tearDown(self) -> None:
        GeConcreteGraph.__call__ = self.call_bak
        GeConcreteGraph.optimize_graph_without_runtime = self.optimize_bak
        return super().tearDown()


    def test_npu_stream_switch(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                add_result = torch.add(in1, in2)
                with torchair.scope.npu_stream_switch('1', 3): 
                    torchair.scope.npu_wait_tensor(in4, add_result)
                    mm_result = torch.mm(in3, in4)
                return add_result, mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                identity_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                    if 'Identity' in op.name:
                        identity_op = op
                stream_label = mm_op.attr["_user_stream_label"].s
                stream_priority = mm_op.attr["_user_stream_priority"].s
                self.assertTrue(stream_label == b'1')
                self.assertTrue(stream_priority == b'3')
                has_control_side = False
                for input_name in identity_op.input:
                    if 'Add' in input_name:
                        has_control_side = True
                self.assertTrue(has_control_side == True)
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        in3 = torch.randn(2, 2)
        in4 = torch.randn(2, 2)
        model(in1, in2, in3, in4)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_super_kernel_scope(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2):
                with torchair.scope.super_kernel('test_scope', 'test_option'): 
                    mm_result = torch.mm(in1, in2)
                return mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                scope = mm_op.attr["_super_kernel_scope"].s
                options = mm_op.attr["_super_kernel_options"].s
                self.assertTrue(scope == b'test_scope')
                self.assertTrue(options == b'test_option')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        model(in1, in2)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_super_kernel_scope_none(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2):
                with torchair.scope.super_kernel('test_scope', 'test_option'): 
                    mm_result = torch.mm(in1, in2)
                    with torchair.scope.super_kernel(None):
                        add_result = torch.add(in1, in2)
                with torchair.scope.super_kernel(None):
                    sub_result = torch.sub(in1, in2)
                return mm_result, add_result, sub_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                add_op = None
                sub_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                    if 'Add' in op.name:
                        add_op = op
                    if 'Sub' in op.name:
                        sub_op = op
                mm_scope = mm_op.attr["_super_kernel_scope"].s
                mm_options = mm_op.attr["_super_kernel_options"].s
                self.assertEqual(mm_scope, b'test_scope')
                self.assertEqual(mm_options, b'test_option')
                add_scope = add_op.attr["_super_kernel_scope"].s
                add_options = add_op.attr["_super_kernel_options"].s
                self.assertEqual(add_scope, b'')
                self.assertEqual(add_options, b'')
                sub_scope = sub_op.attr["_super_kernel_scope"].s
                sub_options = sub_op.attr["_super_kernel_options"].s
                self.assertEqual(sub_scope, b'')
                self.assertEqual(sub_options, b'')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        model(in1, in2)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_super_core_num(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2):
                with torchair.scope.limit_core_num('1', '2'): 
                    mm_result = torch.mm(in1, in2)
                return mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                attr1 = mm_op.attr["_op_aicore_num"].s
                attr2 = mm_op.attr["_op_vectorcore_num"].s
                self.assertTrue(attr1 == b'1')
                self.assertTrue(attr2 == b'2')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        model(in1, in2)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_nested_scope(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                with torchair.scope.super_kernel('1', '2'): 
                    mm_result = torch.mm(in3, in4)
                    with torchair.scope.npu_stream_switch('5', 6): 
                        add_result = torch.add(in1, in2)
                return add_result, mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                add_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                    if 'Add' in op.name:
                        add_op = op
                mm_attr1 = mm_op.attr["_super_kernel_scope"].s
                mm_attr2 = mm_op.attr["_super_kernel_options"].s
                self.assertTrue(mm_attr1 == b'1')
                self.assertTrue(mm_attr2 == b'2')
                add_attr1 = add_op.attr["_super_kernel_scope"].s
                add_attr2 = add_op.attr["_super_kernel_options"].s
                add_attr3 = add_op.attr["_user_stream_label"].s
                add_attr4 = add_op.attr["_user_stream_priority"].s
                self.assertTrue(add_attr1 == b'1')
                self.assertTrue(add_attr2 == b'2')
                self.assertTrue(add_attr3 == b'5')
                self.assertTrue(add_attr4 == b'6')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        in3 = torch.randn(2, 2)
        in4 = torch.randn(2, 2)
        model(in1, in2, in3, in4)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_nested_core_num(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                with torchair.scope.limit_core_num(1, 2):
                    mm_result = torch.mm(in3, in4)
                    with torchair.scope.limit_core_num(3, 4):
                        add_result = torch.add(in1, in2)
                return add_result, mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                add_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                    if 'Add' in op.name:
                        add_op = op
                mm_attr1 = mm_op.attr["_op_aicore_num"].s
                mm_attr2 = mm_op.attr["_op_vectorcore_num"].s
                self.assertTrue(mm_attr1 == b'1')
                self.assertTrue(mm_attr2 == b'2')
                add_attr1 = add_op.attr["_op_aicore_num"].s
                add_attr2 = add_op.attr["_op_vectorcore_num"].s
                self.assertTrue(add_attr1 == b'3')
                self.assertTrue(add_attr2 == b'4')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        in3 = torch.randn(2, 2)
        in4 = torch.randn(2, 2)
        model(in1, in2, in3, in4)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_tranpose(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3):
                abs_01 = torch.abs(in1)
                transpose_01 = torch.transpose(abs_01, 1, 2)
                with torchair.scope.npu_stream_switch("test", 1):
                    matmul_01 = torch.bmm(transpose_01, in1)
                    transpose_02 = torch.transpose(in2, 0, 1)
                    transpose_03 = torch.transpose(in3, 0, 1)
                    matmul_02 = torch.mm(transpose_02, transpose_03)
                    transpose_04 = torch.transpose(abs_01, 1, 2)
                transpose_05 = torch.transpose(abs_01, 1, 2)
                matmul_03 = torch.bmm(transpose_05, in1)
                return matmul_01, matmul_02, matmul_03, transpose_04

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                transpose_list = []
                for op in graph.op:
                    if 'Transpose' in op.name:
                        transpose_list.append(op)
                transpose_01 = transpose_list[0]
                transpose_01_attr1 = transpose_01.attr["_user_stream_label"].s
                transpose_01_attr2 = transpose_01.attr["_user_stream_priority"].s
                self.assertTrue(transpose_01_attr1 == b'')
                self.assertTrue(transpose_01_attr2 == b'')
                transpose_02 = transpose_list[1]
                transpose_02_attr1 = transpose_02.attr["_user_stream_label"].s
                transpose_02_attr2 = transpose_02.attr["_user_stream_priority"].s
                self.assertTrue(transpose_02_attr1 == b'test')
                self.assertTrue(transpose_02_attr2 == b'1')
                transpose_04 = transpose_list[3]
                transpose_04_attr1 = transpose_04.attr["_user_stream_label"].s
                transpose_04_attr2 = transpose_04.attr["_user_stream_priority"].s
                self.assertTrue(transpose_04_attr1 == b'test')
                self.assertTrue(transpose_04_attr2 == b'1')
                transpose_05 = transpose_list[4]
                transpose_05_attr1 = transpose_05.attr["_user_stream_label"].s
                transpose_05_attr2 = transpose_05.attr["_user_stream_priority"].s
                self.assertTrue(transpose_05_attr1 == b'')
                self.assertTrue(transpose_05_attr2 == b'')
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(1024, 1024, 10, dtype=torch.float32)
        in2 = torch.randn(1024, 64, dtype=torch.float32)
        in3 = torch.randn(1024, 1024, dtype=torch.float32)
        model(in1, in2, in3)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_never_timeout_scope(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                with torchair.scope.op_never_timeout(enable=True):
                    mm_result = torch.mm(in1, in2)
                    with torchair.scope.op_never_timeout(enable=False):
                        add_result = torch.add(in3, in4)
                return add_result, mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                add_op = None
                for op in graph.op:
                    if 'MatMul' in op.name:
                        mm_op = op
                    if 'Add' in op.name:
                        add_op = op
                mm_attr = mm_op.attr["_op_exec_never_timeout"].b
                self.assertTrue(mm_attr)
                add_attr = add_op.attr["_op_exec_never_timeout"].b
                self.assertFalse(add_attr)
                ret = func(*args, **kwargs)
                return ret
            return wrapper
        bak_optimization = GeConcreteGraph.optimize_graph_without_runtime
        GeConcreteGraph.optimize_graph_without_runtime = wrapper_call(GeConcreteGraph.optimize_graph_without_runtime)
        model = Model()
        config_view = CompilerConfig()
        npu_backend_view = torchair.get_npu_backend(compiler_config=config_view)
        model = torch.compile(model, backend=npu_backend_view, dynamic=False)
        in1 = torch.randn(2, 2)
        in2 = torch.randn(2, 2)
        in3 = torch.randn(2, 2)
        in4 = torch.randn(2, 2)
        model(in1, in2, in3, in4)
        GeConcreteGraph.optimize_graph_without_runtime = bak_optimization


    def test_never_timeout_scope_invalid_arg(self):
        with self.assertRaises(TypeError) as cm:
            with torchair.scope.op_never_timeout(enable=1):  
                pass  
        self.assertTrue(
            str(cm.exception).startswith("op_never_timeout() argument 'enable' must be bool"),
            f"Unexpected error message: {cm.exception}"
        )


    def test_ge_data_dump(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, data0):
                add_01 = torch.add(data0, data0)
                with torchair.scope.data_dump():
                    add_02 = torch.add(data0, data0)
                return add_01, add_02
            
        input0 = torch.randn(2, 2)
        config = torchair.CompilerConfig()
        config.dump_config.enable_dump = True
        config.dump_config.dump_layer = " Test1 Test2 Test3 "
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, backend=npu_backend, dynamic=False)
        npu_mode(input0)

        dump_dir = Path.cwd() / "ge_dump_data_json"
        matched_files = []

        if dump_dir.exists():
            matched_files = sorted(
                dump_dir.glob("dump_options_*.json"),
                key=lambda p: p.stat().st_mtime,
            )

        assert matched_files, f"No dump_options_*.json found in {dump_dir}"
        
        latest = matched_files[-1]
        with latest.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        found_layers = set()
        dump_obj = data.get("dump", {})
        dump_list = dump_obj.get("dump_list", [])
        for entry in dump_list:
            layers = entry.get("layer")
            if isinstance(layers, list):
                for l in layers:
                    if isinstance(l, str):
                        found_layers.add(l)
        
        expected = {"Test1", "Test2", "Test3"}
        
        missing = expected - found_layers
        assert not missing, f"Missing expected dump layers in {latest}: {missing}. Found layers: {sorted(found_layers)}"


if __name__ == '__main__':
    unittest.main()
