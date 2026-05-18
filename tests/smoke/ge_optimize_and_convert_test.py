import contextlib
import sys
import unittest

import torch
import torch_npu

from torch.export import export, ExportedProgram
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
from torchair._ge_concrete_graph.auto_functionalized_v2 import (
    conveter_auto_functionalize_v2,
    _extract_schema_info,
    recursive_to_fake
)
from torchair.ge.ge_graph_ascend import _NodeIR, _GeGraphAscend
from torchair.ge.ge_graph_ascend import _optimize_and_convert
from torchair.configs.compiler_config import CompilerConfig
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef
from torchair.ge._ge_graph import Tensor

torch_npu.npu.set_device(0)


@contextlib.contextmanager
def _npu_executor_as_default():
    try:
        sys.modules['torch_npu'] = torch_npu
        yield
    finally:
        del sys.modules['torch_npu']


class GeOptimizeAndConvertTest(unittest.TestCase):
    """optimize_and_convert API 及核心组件集成测试"""

    def test_ge_graph_ascend_data_encapsulation(self):
        """测试 _GeGraphAscend 正确封装转换结果"""
        proto = GraphDef()
        ascend_ir_map = {
            "add_node": _NodeIR(source="converter", ops=[], mapping={})
        }

        ge_graph = _GeGraphAscend(
            proto=proto,
            ascend_ir_map=ascend_ir_map,
            original=None,
            optimized=None
        )

        self.assertEqual(ge_graph.proto, proto)
        self.assertEqual(ge_graph.ascend_ir_map, ascend_ir_map)
        self.assertIsNone(ge_graph.original)
        self.assertIsNone(ge_graph.optimized)

    def test_ge_concrete_graph_record_ascend_ir(self):
        """测试 record_ascend_ir 参数正确处理"""
        graph_false = GeConcreteGraph(CompilerConfig())
        self.assertEqual(graph_false._record_asc_ir, False)
        self.assertIsNone(graph_false._ascend_ir_map)
        self.assertEqual(graph_false.get_ascend_ir_map(), {})

        graph_true = GeConcreteGraph(CompilerConfig(), record_ascend_ir=True)
        self.assertEqual(graph_true._record_asc_ir, True)
        self.assertEqual(graph_true._ascend_ir_map, {})

    def test_parse_node_internal_logic(self):
        """测试 _parse_node_internal 保持原逻辑"""
        with _npu_executor_as_default():
            graph = GeConcreteGraph(CompilerConfig())

            x = torch.randn(2, 3).npu()
            y = torch.randn(2, 3).npu()

            ge_outputs = graph._parse_node_internal(
                target=torch.ops.aten.add.Tensor,
                args=(x, y),
                kwargs={},
                meta_outputs=x + y
            )

            self.assertIsNotNone(ge_outputs)

    def test_create_mock_node_for_op(self):
        """测试 mock node 创建"""
        with _npu_executor_as_default():
            graph = GeConcreteGraph(CompilerConfig(), record_ascend_ir=True)

            target = torch.ops.aten.add.Tensor
            mock_node = graph._create_mock_node_for_op(target, [], {})

            self.assertIsNotNone(mock_node)
            self.assertTrue(hasattr(mock_node, 'node'))

    def test_record_ascend_ir(self):
        """测试 ascend_ir 记录功能"""
        with _npu_executor_as_default():
            graph = GeConcreteGraph(CompilerConfig(), record_ascend_ir=True)

            op_def = OpDef()
            op_def.name = "test_op"
            op_def.type = "Add"
            op_def.output_desc.add().name = "y"

            ge_outputs = Tensor(op_def, 0)

            graph._record_ascend_ir(
                "test_node",
                ge_outputs,
                [op_def],
                is_mock=False
            )

            self.assertIn("test_node", graph._ascend_ir_map)
            node_ir = graph._ascend_ir_map["test_node"]
            self.assertEqual(node_ir.source, "converter")
            self.assertEqual(len(node_ir.ops), 1)

    def test_post_process_dict_bytes_conversion(self):
        """测试 bytes → hex 转换"""
        with _npu_executor_as_default():
            graph = GeConcreteGraph(CompilerConfig(), record_ascend_ir=True)

            test_bytes = b"\x01\x02\x03"
            result = graph._post_process_dict(test_bytes)
            self.assertEqual(result, "010203")

            test_data = {
                "level1": {
                    "level2": {
                        "bytes_field": b"\x01\x02"
                    }
                }
            }
            result = graph._post_process_dict(test_data)
            self.assertEqual(result["level1"]["level2"]["bytes_field"], "0102")

    def test_ge_concrete_graph_backward_compatibility(self):
        """测试向后兼容性"""
        graph = GeConcreteGraph(CompilerConfig())

        self.assertEqual(graph._record_asc_ir, False)
        self.assertIsNone(graph._ascend_ir_map)

    def test_optimize_and_convert_simple_model(self):
        """测试简单模型的完整转换流程"""
        with _npu_executor_as_default():
            class SimpleModel(torch.nn.Module):
                def forward(self, x, y):
                    return torch.add(x, y)

            model = SimpleModel()
            x = torch.randn(2, 3).npu()
            y = torch.randn(2, 3).npu()

            ep = export(model, (x, y))
            result = _optimize_and_convert(ep, config=CompilerConfig())

            self.assertIsInstance(result, _GeGraphAscend)
            self.assertIsNotNone(result.proto)
            self.assertIsNotNone(result.ascend_ir_map)
            self.assertTrue(len(result.proto.op) > 0)
            self.assertIsNotNone(result.original)
            self.assertIsNotNone(result.optimized)

    def test_optimize_and_convert_ascend_ir_map_content(self):
        """验证 ascend_ir_map 正确记录节点"""
        with _npu_executor_as_default():
            class AddModel(torch.nn.Module):
                def forward(self, x, y):
                    return torch.add(x, y)

            model = AddModel()
            x = torch.randn(2, 3).npu()
            y = torch.randn(2, 3).npu()

            ep = export(model, (x, y))
            result = _optimize_and_convert(ep)

            self.assertTrue(len(result.ascend_ir_map) > 0)

            for node_name, node_ir in result.ascend_ir_map.items():
                self.assertIsInstance(node_ir, _NodeIR)
                self.assertIn(node_ir.source, ["converter", ""])

    def test_optimize_and_convert_with_custom_decompositions(self):
        """测试 custom_decompositions 参数"""
        with _npu_executor_as_default():
            from torch.export import default_decompositions

            class MatMulModel(torch.nn.Module):
                def forward(self, x, y):
                    return torch.matmul(x, y)

            model = MatMulModel()
            x = torch.randn(2, 3).npu()
            y = torch.randn(3, 4).npu()

            ep = export(model, (x, y))

            custom_decomp = {}
            if torch.ops.aten.matmul.default in default_decompositions():
                custom_decomp[torch.ops.aten.matmul.default] = default_decompositions()[torch.ops.aten.matmul.default]

            result = _optimize_and_convert(ep, custom_decompositions=custom_decomp)

            self.assertIsNotNone(result.optimized)

    def test_optimize_and_convert_multi_layer_model(self):
        """测试多层模型的转换"""
        with _npu_executor_as_default():
            class MultiLayerModel(torch.nn.Module):
                def forward(self, x):
                    x = torch.add(x, 1)
                    x = torch.mul(x, 2)
                    x = torch.sub(x, 3)
                    return x

            model = MultiLayerModel()
            x = torch.randn(2, 3).npu()

            ep = export(model, (x,))
            result = _optimize_and_convert(ep)

            self.assertTrue(len(result.ascend_ir_map) >= 1)

    def test_optimize_and_convert_model_with_parameters(self):
        """测试带参数的模型转换"""
        with _npu_executor_as_default():
            class LinearModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.randn(3, 4))

                def forward(self, x):
                    return torch.matmul(x, self.weight)

            model = LinearModel().npu()
            x = torch.randn(2, 3).npu()

            ep = export(model, (x,))
            result = _optimize_and_convert(ep)

            self.assertIsNotNone(result.proto)

    def test_optimize_and_convert_conv2d_batchnorm(self):
        """测试 Conv2d 和 BatchNorm 模型"""
        with _npu_executor_as_default():
            class Conv2dModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 6, 3)

                def forward(self, x):
                    return self.conv(x)

            model = Conv2dModel().npu()
            x = torch.randn(1, 3, 32, 32).npu()

            ep = export(model, (x,))
            result = _optimize_and_convert(ep)
            self.assertIsNotNone(result.proto)

            class BatchNormModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bn = torch.nn.BatchNorm2d(6)

                def forward(self, x):
                    return self.bn(x)

            model_bn = BatchNormModel().npu()
            x_bn = torch.randn(1, 6, 32, 32).npu()

            ep_bn = export(model_bn, (x_bn,))
            result_bn = _optimize_and_convert(ep_bn)
            self.assertIsNotNone(result_bn.proto)

    def test_optimize_and_convert_activations(self):
        """测试激活函数模型（ReLU、Sigmoid、Softmax）"""
        with _npu_executor_as_default():
            class ReLUModel(torch.nn.Module):
                def forward(self, x):
                    return torch.relu(x)

            model_relu = ReLUModel()
            x = torch.randn(2, 3).npu()

            ep_relu = export(model_relu, (x,))
            result_relu = _optimize_and_convert(ep_relu)
            self.assertIsNotNone(result_relu.proto)

            class SigmoidModel(torch.nn.Module):
                def forward(self, x):
                    return torch.sigmoid(x)

            model_sigmoid = SigmoidModel()
            ep_sigmoid = export(model_sigmoid, (x,))
            result_sigmoid = _optimize_and_convert(ep_sigmoid)
            self.assertIsNotNone(result_sigmoid.proto)

            class SoftmaxModel(torch.nn.Module):
                def forward(self, x):
                    return torch.softmax(x, dim=-1)

            model_softmax = SoftmaxModel()
            ep_softmax = export(model_softmax, (x,))
            result_softmax = _optimize_and_convert(ep_softmax)
            self.assertIsNotNone(result_softmax.proto)

    def test_optimize_and_convert_special_ops(self):
        """测试特殊操作（Transpose、Mean、Element-wise）"""
        with _npu_executor_as_default():
            class TransposeModel(torch.nn.Module):
                def forward(self, x):
                    return torch.transpose(x, 0, 1)

            model_transpose = TransposeModel()
            x = torch.randn(3, 4).npu()

            ep_transpose = export(model_transpose, (x,))
            result_transpose = _optimize_and_convert(ep_transpose)
            self.assertIsNotNone(result_transpose.proto)

            class MeanModel(torch.nn.Module):
                def forward(self, x):
                    return torch.mean(x, dim=1)

            model_mean = MeanModel()
            x_mean = torch.randn(2, 3, 4).npu()

            ep_mean = export(model_mean, (x_mean,))
            result_mean = _optimize_and_convert(ep_mean)
            self.assertIsNotNone(result_mean.proto)

            class ElementWiseModel(torch.nn.Module):
                def forward(self, x, y):
                    a = torch.add(x, y)
                    b = torch.sub(x, y)
                    c = torch.mul(a, b)
                    return torch.div(c, 2.0)

            model_elem = ElementWiseModel()
            x_elem = torch.randn(2, 3).npu()
            y_elem = torch.randn(2, 3).npu()

            ep_elem = export(model_elem, (x_elem, y_elem))
            result_elem = _optimize_and_convert(ep_elem)
            self.assertIsNotNone(result_elem.proto)

    def test_optimize_and_convert_with_config(self):
        """测试 CompilerConfig 参数"""
        with _npu_executor_as_default():
            class SimpleModel(torch.nn.Module):
                def forward(self, x):
                    return x + 1

            model = SimpleModel()
            x = torch.randn(2, 3).npu()

            ep = export(model, (x,))

            config = CompilerConfig()
            config.experimental_config.remove_noop_ops = True

            result = _optimize_and_convert(ep, config=config)

            self.assertIsNotNone(result.proto)


if __name__ == "__main__":
    unittest.main()
