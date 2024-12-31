import os
import time
import logging

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
                with torchair.NpuStreamSwitch('1', 3): 
                    torchair.ops.npu_wait_tensor(in4, add_result)
                    mm_result = torch.mm(in3, in4)
                return add_result, mm_result

        def wrapper_call(func):
            def wrapper(*args, **kwargs):
                assert len(args) > 0
                graph = args[0].graph
                mm_op = None
                identity_op = None
                for op in graph.op:
                    if op.name == 'MatMul':
                        mm_op = op
                    if op.name == 'Identity':
                        identity_op = op
                stream_label = mm_op.attr["_user_stream_label"].s
                stream_priority = mm_op.attr["_user_stream_priority"].s
                self.assertTrue(stream_label == b'1')
                self.assertTrue(stream_priority == b'3')
                has_control_side = False
                for input_name in identity_op.input:
                    if input_name == 'Add:-1':
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


if __name__ == '__main__':
    unittest.main()
