import os
os.environ['TNG_LOG_LEVEL'] = '0'

import torch
import torchair
from torchair.ge._ge_graph import GeGraph
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)

import unittest
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

m = torch.library.Library("npu", "FRAGMENT")
m.define("my_inplace_auto_kwargs_str(Tensor(a!) x, Tensor y, *, str alpha='alpha') -> ()")

@torch.library.impl(m, "my_inplace_auto_kwargs_str", "Meta")
def my_inplace_meta(x, y, alpha='alpha'):
    pass

class TestCustomOps(unittest.TestCase):
    def setUp(self) -> None:
        self.call_bak = GeConcreteGraph.__call__
        return super().setUp()

    def tearDown(self) -> None:
        GeConcreteGraph.__call__ = self.call_bak
        return super().tearDown()

    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_multi_ops_in_converter(self):
        m.define("multi_ops_in_converter(Tensor(a!) x, Tensor y) -> Tensor")


        @torch.library.impl(m, "multi_ops_in_converter", "Meta")
        def my_inplace_meta(x, y):
            return torch.empty_like(y)        

        @torchair.register_fx_node_ge_converter(torch.ops.npu.multi_ops_in_converter.default)        
        def converter_multi_ops_in_converter(x, y, meta_outputs = None):
            tmp = torchair.ge.custom_op(                  
                "MultiInplace1",
                inputs={
                    "x": x,
                    "y": y,
                },
                outputs=['x', 'z']
            )
            out = torchair.ge.custom_op(                  
                "MultiInplace2",
                inputs={
                    "x": tmp[0],
                    "p": tmp[1],
                },
                outputs=['x', 'q']
            )
            return out[1]

        def cus_func(x, y):
            o2 = torch.ops.npu.multi_ops_in_converter(x, y)
            return o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name : op_node.input for op_node in geGraph.op}
                    print(f'op_name_dict is : {op_name_dict}')
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MultiInplace1", op_name_dict)
                    self.assertIn("MultiInplace2", op_name_dict)
                    self.assertIn("Identity", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MultiInplace1"])
                    self.assertIn("MultiInplace1:0", op_name_dict["MultiInplace2"])
                    self.assertIn("MultiInplace2:-1", op_name_dict["Identity"])
                    self.assertIn("TensorMove:0", op_name_dict["Identity"])
                    self.assertIn("Identity:0", op_name_dict["NetOutput"])

                    ret = func(*args, **kwargs)
                    return ret
               
                return wrapper      
            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        with self.assertRaises(RuntimeError) as context:
            out = compile_func(input1, input2)
        self.assertTrue("Assert outputs.empty()" in str(context.exception))

    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_as_stride(self):        
        m.define("my_inplace_auto2(Tensor(a!) x, Tensor y) -> Tensor")

        @torch.library.impl(m, "my_inplace_auto2", "Meta")
        def my_inplace_meta(x, y):
            return torch.empty_like(y)        

        @torchair.register_fx_node_ge_converter(torch.ops.npu.my_inplace_auto2.default)        
        def converter_npu_add_custom(x, y, meta_outputs = None):
            out = torchair.ge.custom_op(                  
                "MyInpalceAuto2",
                inputs={
                    "x": x,
                    "y": y,
                },
                outputs=['x', 'z']
            )
            return out[1]

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            slice = add0[:, 1:]
            o2 = torch.ops.npu.my_inplace_auto2(slice, y)
            add1 = torch.add(slice, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name : op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("AsStrided", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["AsStrided"])
                    self.assertIn("MyInpalceAuto2", op_name_dict)
                    self.assertIn("ViewCopy", op_name_dict)
                    self.assertIn("MyInpalceAuto2:-1", op_name_dict["ViewCopy"])
                    self.assertIn("TensorMove:0", op_name_dict["ViewCopy"])
                    self.assertIn("StridedSliceV2", op_name_dict)
                    self.assertIn("ViewCopy:0", op_name_dict["StridedSliceV2"])

                    ret = func(*args, **kwargs)
                    return ret
               
                return wrapper      
            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2) 


    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_not_view(self):
        m.define("my_inplace_auto1(Tensor(a!) x, Tensor y) -> Tensor")

        @torch.library.impl(m, "my_inplace_auto1", "Meta")
        def my_inplace_meta(x, y):
            return torch.empty_like(y)        

        @torchair.register_fx_node_ge_converter(torch.ops.npu.my_inplace_auto1.default)        
        def converter_npu_add_custom(x, y, meta_outputs = None):
            out = torchair.ge.custom_op(                  
                "MyInpalceAuto1",
                inputs={
                    "x": x,
                    "y": y,
                },
                outputs=['x', 'z']
            )
            return out[1]

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto1(add0, y)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name : op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInpalceAuto1", op_name_dict)
                    self.assertIn("MyInpalceAuto1:0", op_name_dict["Add_1"])

                    ret = func(*args, **kwargs)
                    return ret
               
                return wrapper      
            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)


    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_no_output(self):
        m.define("my_inplace_auto_no_output(Tensor(a!) x, Tensor y) -> ()")

        @torch.library.impl(m, "my_inplace_auto_no_output", "Meta")
        def my_inplace_meta(x, y):
            pass

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto_no_output(add0, y)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInplaceAutoNoOutput", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInplaceAutoNoOutput"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)

    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_two_inplace(self):
        m.define("my_two_inplace(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) kv_state, Tensor(b!) score_state,"
        "Tensor ape, Tensor norm_weight, Tensor rope_sin, Tensor rope_cos) -> (Tensor)")

        @torch.library.impl(m, "my_two_inplace", "Meta")
        def my_inplace_meta(x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos):
            return torch.empty_like(x)

        def cus_func(x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_two_inplace(x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    print(f"---> op_name_dict: {op_name_dict}")
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("TensorMove_1", op_name_dict)
                    self.assertIn("MyTwoInplace", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyTwoInplace"])
                    self.assertIn("TensorMove_1:0", op_name_dict["MyTwoInplace"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        input3 = torch.ones(2, 1)
        with self.assertRaises(RuntimeError) as context:
            out = compile_func(input1, input1, input1, input2, input3, input1, input1, input1, input1) 
        self.assertTrue("Assert outputs.empty()" in str(context.exception))

    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_infer_symbol_with_auto_functionalize(self):
        m.define("my_op_inplace_z(Tensor(a!) x, Tensor y) -> Tensor z")

        @torch.library.impl(m, "my_op_inplace_z", "Meta")
        def my_op_meta(x, y):
            size_y_0 = list(y.shape)[0] * 2
            size_y_1 = list(y.shape)[1] // 2
            out = torch.empty((size_y_0, size_y_1), dtype=y.dtype, device=y.device)
            return out

        def cus_func(x, y):
            return torch.ops.npu.my_op_inplace_z(x, y)

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    import json
                    import ast
                    self.assertTrue("MyOpInplaceZ" in [op.name for op in geGraph.op])
                    for op in geGraph.op:
                        if op.name == "MyOpInplaceZ":                                               
                            inference_rule = json.loads(op.attr["_inference_rule"].s)
                            self.assertEqual(inference_rule["shape"]["inputs"][0][0], "s2")
                            self.assertEqual(inference_rule["shape"]["inputs"][0][1], "s3")
                            self.assertEqual(inference_rule["shape"]["inputs"][1][0], "s0")
                            self.assertEqual(inference_rule["shape"]["inputs"][1][1], "s1")
                            
                            is_high_python_version = hasattr(ast, 'unparse')
                            self.assertEqual(inference_rule["shape"]["outputs"][0][0], "s2")
                            self.assertEqual(inference_rule["shape"]["outputs"][0][1], "s3")
                            s2_out = "2 * s0" if is_high_python_version else "(2*s0)"
                            self.assertEqual(inference_rule["shape"]["outputs"][1][0], s2_out)
                            self.assertEqual(inference_rule["shape"]["outputs"][1][1], "Floor(Div(s1, 2))")

                            self.assertEqual(inference_rule["dtype"][0], 3)
                            self.assertEqual(inference_rule["dtype"][1], 0)
                       

                    ret = func(*args, **kwargs)
                    return ret
               
                return wrapper    


            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=True)  
        input1 = torch.ones((4, 4), dtype=torch.int32)
        input2 = torch.ones((4, 4), dtype=torch.float)

        with self.assertRaises(RuntimeError) as context:
            out = compile_func(input1, input2)
        self.assertTrue("Assert outputs.empty()" in str(context.exception))

    def test_auto_functionalize_kwargs_int_with_input(self):
        m.define("my_inplace_auto_kwargs_int(Tensor(a!) x, Tensor y, *, int alpha=1) -> ()")

        @torch.library.impl(m, "my_inplace_auto_kwargs_int", "Meta")
        def my_inplace_meta(x, y, alpha=1):
            pass

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto_kwargs_int(add0, y, alpha=2)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInplaceAutoKwargsInt", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInplaceAutoKwargsInt"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)

    def test_auto_functionalize_kwargs_str(self):        

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto_kwargs_str(add0, y, alpha='beta')
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInplaceAutoKwargsStr", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInplaceAutoKwargsStr"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)
    
    def test_auto_functionalize_kwargs_str_without_input(self):

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto_kwargs_str(add0, y)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInplaceAutoKwargsStr", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInplaceAutoKwargsStr"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)
    
    def test_auto_functionalize_optional_input(self):
        m.define("my_inplace_auto_option_input(Tensor(a!) x, Tensor y, *, Tensor?z=None, str alpha='alpha') -> ()")

        @torch.library.impl(m, "my_inplace_auto_option_input", "Meta")
        def my_inplace_meta(x, y, z=None, alpha='alpha'):
            pass

        def cus_func(x, y):
            add0 = torch.add(x, 1)
            o2 = torch.ops.npu.my_inplace_auto_option_input(add0, y)
            add1 = torch.add(add0, 1)
            return add1, o2

        def warp_concrete_graph():
            def wrapper_call(func):
                def wrapper(*args, **kwargs):
                    assert len(args) > 0
                    geGraph: GeGraph = args[0]._graph
                    op_name_dict = {op_node.name: op_node.input for op_node in geGraph.op}
                    self.assertIn("TensorMove", op_name_dict)
                    self.assertIn("MyInplaceAutoOptionInput", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInplaceAutoOptionInput"])

                    ret = func(*args, **kwargs)
                    return ret

                return wrapper

            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        out = compile_func(input1, input2)        

    def test_auto_converter_no_match_inputs(self):
        m.define("my_op_no_match_inputs(Tensor self, Tensor updates, Tensor indices) -> Tensor")

        @torch.library.impl(m, "my_op_no_match_inputs", "Meta")
        def my_op_no_match_inputs_meta(self, updates, indices):
            return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                x = torch.ops.npu.my_op_no_match_inputs(x, y, z)
                add = torch.add(x, 5)
                return add

        input0 = torch.zeros(2, 2, dtype=torch.float32)
        input1 = torch.randn(2, 2, dtype=torch.float32)
        input2 = torch.randn(2, 2, dtype=torch.float32)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(input0, input1, input2)
        self.assertTrue("Failed to auto converter npu.my_op_no_match_inputs.default to AscendIR: " + \
            "the number of torch tensor inputs does not match the AscendIR MyOpNoMatchInputs inputs, " + \
            "you can check your torch and AscendIR registration or try to implement " + \
            "the converter manually according to the following code." in str(context.exception))

    def test_auto_converter_no_match_attrs(self):
        m.define("my_op_no_match_attrs(Tensor self, Tensor updates, *, int indices=2, int dim=2) -> Tensor")

        @torch.library.impl(m, "my_op_no_match_attrs", "Meta")
        def my_op_no_match_attrs_meta(self, updates, indices=2, dim=2):
            return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = torch.ops.npu.my_op_no_match_attrs(x, y)
                add = torch.add(x, 5)
                return add

        input0 = torch.zeros(2, 2, dtype=torch.float32)
        input1 = torch.randn(2, 2, dtype=torch.float32)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(input0, input1)
        self.assertTrue("Failed to auto converter npu.my_op_no_match_attrs.default to AscendIR: " + \
            "the number of torch non-tensor inputs greater than the AscendIR MyOpNoMatchAttrs attrs, " + \
            "you can check your torch and AscendIR registration or try to implement " + \
            "the converter manually according to the following code." in str(context.exception))

    def test_auto_converter_no_match_outputs(self):
        m.define("my_op_no_match_outputs(Tensor(a!) self, Tensor updates, *, int indices=2) -> Tensor")

        @torch.library.impl(m, "my_op_no_match_outputs", "Meta")
        def my_op_no_match_outputs_meta(self, updates, indices=2):
            return self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = torch.ops.npu.my_op_no_match_outputs(x, y)
                add = torch.add(x, 5)
                return add

        input0 = torch.zeros(2, 2, dtype=torch.float32)
        input1 = torch.randn(2, 2, dtype=torch.float32)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(input0, input1)
        self.assertTrue("Failed to auto converter npu.my_op_no_match_outputs.default to AscendIR: " + \
            "the number of torch outputs does not match the AscendIR MyOpNoMatchOutputs outputs, " + \
            "you can check your torch and AscendIR registration or try to implement " + \
            "the converter manually according to the following code." in str(context.exception))

if __name__ == "__main__":
    unittest.main()
    