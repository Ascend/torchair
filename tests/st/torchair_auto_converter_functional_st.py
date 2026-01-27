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

m = torch.library.Library("npu", "FRAGMENT")

class TestCustomOps(unittest.TestCase):

    @unittest.skipIf(torch.__version__ < "2.2", "torch._auto_functionalize is unsupported when torch < 2.2")
    def test_auto_functionalize_multi_ops_in_converter(self):
        m.define("multi_ops_in_converter(Tensor(a!) x, Tensor y) -> Tensor")


        @torch.library.impl(m, "multi_ops_in_converter", "Meta")
        def my_inplace_meta(x, y):
            return torch.empty_like(y)        

        @torchair.register_fx_node_ge_converter(torch.ops.npu.multi_ops_in_converter.default)        
        def converter_multi_ops_in_converter(x, y, meta_outputs = None):
            tmp = torchair.ge.custom_op(                  
                "MyInpalceAuto1",
                inputs={
                    "x": x,
                    "y": y,
                },
                outputs=['x', 'z']
            )
            out = torchair.ge.custom_op(                  
                "MyInpalceAuto2",
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
                    self.assertIn("MyInpalceAuto1", op_name_dict)
                    self.assertIn("MyInpalceAuto2", op_name_dict)
                    self.assertIn("Identity", op_name_dict)
                    self.assertIn("TensorMove:0", op_name_dict["MyInpalceAuto1"])
                    self.assertIn("MyInpalceAuto1:0", op_name_dict["MyInpalceAuto2"])
                    self.assertIn("MyInpalceAuto2:-1", op_name_dict["Identity"])
                    self.assertIn("TensorMove:0", op_name_dict["Identity"])
                    self.assertIn("Identity:0", op_name_dict["NetOutput"])

                    ret = func(*args, **kwargs)
                    return ret
               
                return wrapper      
            GeConcreteGraph.__call__ = wrapper_call(GeConcreteGraph.__call__)

        warp_concrete_graph()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        compile_func = torch.compile(cus_func, backend=npu_backend, fullgraph=True, dynamic=False)
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 1)
        with self.assertRaises(RuntimeError) as context:
            out = compile_func(input1, input2)
        self.assertTrue("Assert outputs.empty()" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
    