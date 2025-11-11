import logging
import unittest
import os
import unittest.mock
import torch
from torch.utils._mode_utils import no_dispatch
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core._concrete_graph import ConcreteGraphBase
from torchair.npu_fx_compiler import _NpuGraphConverter, _next_unique_graph_id, _NpuFxCompiler
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph as ConcreteGraph
import torchair.ge as ge
from torchair._ge_concrete_graph import ge_apis as raw_ops
from torchair.ge._ge_graph import GeGraph, compat_as_bytes
from torchair._ge_concrete_graph.ge_ir_pb2 import OpDef
from torchair._ge_concrete_graph.compat_ir import is_cann_compat, ge_op, IrDef, IrElement

torchair.logger.setLevel(logging.DEBUG)


class CustomOpSt(unittest.TestCase):
    def test_custom_basic(self):
        with GeGraph():
            data1 = raw_ops.Data(index=0, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data1')
            data2 = raw_ops.Data(index=1, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data2')

            tensor = ge.custom_op('CustomOp', inputs={'x1': data1, 'x2': data2}, outputs=['y'])
            self.assertTrue(isinstance(tensor, raw_ops.Tensor))
            op_def: OpDef = tensor.node

            self.assertEqual(op_def.type, 'CustomOp')
            self.assertEqual(len(op_def.input), 2)
            self.assertEqual(op_def.input[0], 'data1:0')
            self.assertEqual(op_def.input[1], 'data2:0')
            self.assertEqual(op_def.input_desc[0].name, 'x1')
            self.assertEqual(op_def.input_desc[1].name, 'x2')
            self.assertEqual(len(op_def.output_desc), 1)
            self.assertEqual(op_def.output_desc[0].name, 'y')

    def test_custom_not_change_input(self):
        with GeGraph():
            data1 = raw_ops.Data(index=0, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data1')
            data2 = raw_ops.Data(index=1, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data2')

            serialized_desc1 = data1.node.output_desc[0].SerializeToString()
            serialized_desc2 = data2.node.output_desc[0].SerializeToString()

            ge.custom_op('CustomOp', inputs={'x1': data1, 'x2': data2}, outputs=['y'])

            after_called_custom_serialized_desc1 = data1.node.output_desc[0].SerializeToString()
            after_called_custom_serialized_desc2 = data2.node.output_desc[0].SerializeToString()

            self.assertEqual(serialized_desc1, after_called_custom_serialized_desc1)
            self.assertEqual(serialized_desc2, after_called_custom_serialized_desc2)

    def test_custom_attr(self):
        attrs = dict()
        attrs['f'] = ge.attr.Float(1.0)
        attrs['i'] = ge.attr.Int(1)
        attrs['b'] = ge.attr.Bool(True)
        attrs['s'] = ge.attr.Str("test")
        attrs['dt'] = ge.attr.DataType(ge.DataType.DT_FLOAT)
        attrs['list_f'] = ge.attr.ListFloat([1.0, 2.0])
        attrs['list_i'] = ge.attr.ListInt([1, 2])
        attrs['list_b'] = ge.attr.ListBool([True, False])
        attrs['list_s'] = ge.attr.ListStr(["test1", "test2"])
        attrs['list_dt'] = ge.attr.ListDataType([ge.DataType.DT_FLOAT, ge.DataType.DT_INT32])
        attrs['list_list_i'] = ge.attr.ListListInt([[1, 2], [3, 4]])
        attrs['list_list_float'] = ge.attr.ListListFloat([[1.0, 2.0], [3.0, 4.0]])

        with GeGraph():
            tensor = ge.custom_op('CustomOp', inputs=None, outputs=['y'], attrs=attrs)
        self.assertTrue(isinstance(tensor, raw_ops.Tensor))
        op_def: OpDef = tensor.node

        self.assertEqual(len(op_def.attr), len(attrs))
        for k, v in op_def.attr.items():
            self.assertTrue(k in attrs)
            self.assertEqual(attrs[k].get(v), attrs.get(k))

    def test_custom_with_dynamic_input(self):
        with GeGraph():
            data = raw_ops.Data(index=0, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data')
            tensor = ge.custom_op('CustomOp', inputs={'x': [data] * 3}, outputs=['y'])
        op_def: OpDef = tensor.node

        self.assertEqual(op_def.type, 'CustomOp')
        self.assertEqual(len(op_def.input), 3)
        self.assertEqual(op_def.input[0], 'data:0')
        self.assertEqual(op_def.input[1], 'data:0')
        self.assertEqual(op_def.input[2], 'data:0')
        self.assertEqual(op_def.input_desc[0].name, 'x0')
        self.assertEqual(op_def.input_desc[1].name, 'x1')
        self.assertEqual(op_def.input_desc[2].name, 'x2')

    def test_custom_with_optional_input(self):
        with GeGraph():
            tensor = ge.custom_op('CustomOp', inputs={'x': None}, outputs=['y'])
        op_def: OpDef = tensor.node

        self.assertEqual(op_def.type, 'CustomOp')
        self.assertEqual(len(op_def.input), 1)
        self.assertEqual(len(op_def.input_desc), 1)
        self.assertEqual(op_def.input_desc[0].name, 'x')

    def test_custom_with_dynamic_output(self):
        with GeGraph():
            tensors = ge.custom_op('CustomOp', inputs=None, outputs=['x', ('y', 1), ('z', 2)])
        self.assertTrue(isinstance(tensors, tuple))
        self.assertEqual(len(tensors), 3)
        self.assertTrue(isinstance(tensors[0], raw_ops.Tensor))
        self.assertTrue(isinstance(tensors[1], list))
        self.assertEqual(len(tensors[1]), 1)
        self.assertTrue(isinstance(tensors[1][0], raw_ops.Tensor))
        self.assertTrue(isinstance(tensors[2], list))
        self.assertEqual(len(tensors[2]), 2)
        self.assertTrue(isinstance(tensors[2][0], raw_ops.Tensor))
        self.assertTrue(isinstance(tensors[2][1], raw_ops.Tensor))

        op_def: OpDef = tensors[0].node

        self.assertEqual(op_def.type, 'CustomOp')
        self.assertEqual(len(op_def.output_desc), 4)
        self.assertEqual(op_def.output_desc[0].name, 'x')
        self.assertEqual(op_def.output_desc[1].name, 'y0')
        self.assertEqual(op_def.output_desc[2].name, 'z0')
        self.assertEqual(op_def.output_desc[3].name, 'z1')

    def test_custom_with_no_output(self):
        with GeGraph():
            tensors = ge.custom_op('CustomOp', inputs=None, outputs=None)
        self.assertTrue(tensors is None)

    def test_custom_with_no_input(self):
        with GeGraph():
            tensor = ge.custom_op('CustomOp', inputs=None, outputs=['y'])
        op_def: OpDef = tensor.node
        self.assertEqual(len(op_def.input), 0)
        self.assertEqual(len(op_def.input_desc), 0)

    def test_custom_with_invalid_input(self):
        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs=3, outputs=None)

        self.assertEqual(str(cm.exception),
                         "Invalid input type:int vs expect one of [None, dict] for custom op 'CustomOp'.")

        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs={'x': 3}, outputs=None)

        self.assertEqual(str(cm.exception),
                         "Invalid input 'x' type:int vs expect one of [None, ge.Tensor, list[ge.Tensor]] "
                         "for custom op 'CustomOp'.")

        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs={'x': [3]}, outputs=None)

        self.assertEqual(str(cm.exception),
                         "Invalid input 'x'[0] type:int vs expect 'ge.Tensor' for custom op 'CustomOp'.")

    def test_custom_with_invalid_attr(self):
        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', attrs={'x': 3}, inputs=None, outputs=None)

        self.assertEqual(str(cm.exception),
                         "Invalid attr 'x' type:int vs expect one of [ge.attr.Bool, ge.attr.DataType, "
                         "ge.attr.Float, ge.attr.Int, ge.attr.ListBool, ge.attr.ListDataType, "
                         "ge.attr.ListFloat, ge.attr.ListInt, ge.attr.ListListFloat, ge.attr.ListListInt, "
                         "ge.attr.ListStr, ge.attr.Str] for custom op 'CustomOp'.")

    def test_custom_with_invalid_output(self):
        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs=None, outputs=3)

        self.assertEqual(str(cm.exception),
                         "Invalid output type:int vs expect one of [list, tuple] for custom "
                         "op 'CustomOp'.")

        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs=None, outputs=[3])

        self.assertEqual(str(cm.exception),
                         "Invalid output type:int vs expect one of [str, tuple[str, int]] for custom "
                         "op 'CustomOp'.")

        with self.assertRaises(AssertionError) as cm, GeGraph():
            ge.custom_op('CustomOp', inputs=None, outputs=[('y',)])

        self.assertEqual(str(cm.exception),
                         "Invalid output type:tuple[str] vs expect tuple[str, int] for custom "
                         "op 'CustomOp'.")

    def test_public_api_register_fx_node_ge_converter(self):
        from torchair import register_fx_node_ge_converter
        import inspect
        from torch.library import Library, impl
        m = Library("test", "DEF")
        m.define("custom_op(Tensor input1) -> Tensor")

        @impl(m, "custom_op", "Meta")
        def custom_op_meta(x, y):
            return torch.empty_like(x)

        from torchair._ge_concrete_graph.fx2ge_converter import Converter, _wrap_converter
        test_add_opp = []
        def my_warp_fun(self, converter):
            wrapped_converter = _wrap_converter(converter)
            if 'meta_outputs' in inspect.signature(converter).parameters:
                wrapped_converter.require_meta = True
            else:
                wrapped_converter.require_meta = False
            try:
                self._aten_op._ge_converter = wrapped_converter
                # 记录个UT 可观测的信息
                test_add_opp.append(self._aten_op)
            except:
                global _CONVERTERS
                _CONVERTERS.update({self._aten_op: wrapped_converter})
            return self

        with unittest.mock.patch.object(Converter, '__call__', my_warp_fun):
            @register_fx_node_ge_converter(torch.ops.test.custom_op.default)
            def conveter_custom_op(input1, out, meta_outputs):
                return input1

        self.assertTrue(torch.ops.test.custom_op.default in test_add_opp)

    def test_Const_Cast(self):

        with GeGraph():
            const = torchair.ge.Const(10, dtype=ge.DataType.DT_FLOAT, node_name='Const_name', readable=True)
            data = raw_ops.Data(index=0, dtype=ge.DataType.DT_INT32, placement='CPU', node_name='data')
            cast = torchair.ge.Cast(data, dst_type=ge.DataType.DT_FLOAT, node_name='cast_name')

            self.assertEqual(const.node.type, 'Const')
            self.assertEqual(const.node.name, 'Const_name')
            self.assertEqual(const.node.attr["_readable_value"].s, compat_as_bytes(f"{repr(10)}"))
            self.assertEqual(cast.node.type, 'Cast')
            self.assertEqual(cast.node.name, 'cast_name')
            self.assertEqual(cast.node.attr["dst_type"].i, ge.DataType.DT_FLOAT)

    def test_public_Class(self):
        self.assertEqual(len(torchair.ge.Tensor.__dict__.keys()), 8)
        self.assertIn('__init__', torchair.ge.Tensor.__dict__.keys())
        self.assertIn('index', torchair.ge.Tensor.__dict__.keys())
        self.assertIn('dtype', torchair.ge.Tensor.__dict__.keys())
        self.assertIn('rank', torchair.ge.Tensor.__dict__.keys())
        self.assertEqual(len(torchair.ge.TensorSpec.__dict__.keys()), 8)
        self.assertIn('__init__', torchair.ge.TensorSpec.__dict__.keys())
        self.assertIn('dtype', torchair.ge.TensorSpec.__dict__.keys())
        self.assertIn('rank', torchair.ge.TensorSpec.__dict__.keys())
        self.assertIn('size', torchair.ge.TensorSpec.__dict__.keys())
        # 枚举值考虑扩展不做长度校验，取典型值保证接口功能
        self.assertIn('FORMAT_FRACTAL_NZ', dir(torchair.ge.Format))
        self.assertIn('DT_BF16', dir(torchair.ge.DataType))

    def test_is_cann_compat_success(self):
        ir_v1 = IrDef("TestV1") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        ir_v2 = IrDef("TestV2") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .optional_input("x4", "DT_FLOAT16, DT_BOOL, DT_FLOAT32") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .attr("attr3", ge.attr.Str("BSH")) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        # V1是老cann的能力，V2是新cann的能力
        # 老PTA + 老cann
        success = is_cann_compat(optype="TestV1",
                                 runtime_optional_inputs=("x3",),
                                 runtime_optional_attrs=("attr2",))
        self.assertEqual(success, "")

        # 新PTA + 新cann
        success = is_cann_compat(optype="TestV2",
                                 runtime_optional_inputs=("x3", "x4"),
                                 runtime_optional_attrs=("attr2", "attr3"))
        self.assertEqual(success, "")

        # 老PTA + 新cann
        success = is_cann_compat(optype="TestV2",
                                 runtime_optional_inputs=("x3",),
                                 runtime_optional_attrs=("attr2",))
        self.assertEqual(success, "")

        # 新PTA + 老cann, 可选输入不传，新增optional_attr为默认值(因此已被滤除，所以输入没有attr3)
        success = is_cann_compat(optype="TestV1",
                                 runtime_optional_inputs=("x3",),
                                 runtime_optional_attrs=("attr2",))
        self.assertEqual(success, "")

    def test_is_cann_compat_assert_raise(self):
        ir_v1 = IrDef("TestV1") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        # V1是老cann的能力，V2是新cann的能力
        # 使用不支持的optype
        error_msg = is_cann_compat(
            optype="TestV100", runtime_optional_inputs=(), runtime_optional_attrs=())

        self.assertEqual(error_msg,
                         f"OperatorFactory find optype TestV100 failed, "
                         f"maybe you need upgrade cann version.")

        # 新PTA + 老cann，可选输入不为空
        error_msg = is_cann_compat(optype="TestV1",
                                   runtime_optional_inputs=("x3", "x4"),
                                   runtime_optional_attrs=("attr2",))

        self.assertEqual(error_msg,
                         f"optype TestV1 unsupport optional input [x4], optional attr [], "
                         f"please upgrade cann version.")

        # 新PTA + 老cann， 可选attr不是默认值
        error_msg = is_cann_compat(optype="TestV1",
                                   runtime_optional_inputs=("x3",),
                                   runtime_optional_attrs=("attr2", "attr3"))

        self.assertEqual(error_msg,
                         f"optype TestV1 unsupport optional input [], optional attr [attr3], "
                         f"please upgrade cann version.")

    def test_ir_def(self):
        ir = IrDef("TestV1") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        self.assertEqual("TestV1", ir.op_type)
        self.assertEqual(IrElement.RequireInput, ir.indexed_inputs[0][0])
        self.assertEqual("x1", ir.indexed_inputs[0][1])
        self.assertEqual(
            "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8", ir.indexed_inputs[0][2])

        self.assertEqual(IrElement.DynamicInput, ir.indexed_inputs[1][0])
        self.assertEqual("x2", ir.indexed_inputs[1][1])
        self.assertEqual(
            "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8", ir.indexed_inputs[1][2])

        self.assertEqual(IrElement.OptionalInput, ir.indexed_inputs[2][0])
        self.assertEqual("x3", ir.indexed_inputs[2][1])
        self.assertEqual("DT_FLOAT16, DT_BF16", ir.indexed_inputs[2][2])

        self.assertEqual(IrElement.RequireAttr, ir.indexed_attrs[0][0])
        self.assertEqual("attr1", ir.indexed_attrs[0][1])
        self.assertEqual(ge.attr.Int, ir.indexed_attrs[0][2])

        self.assertEqual(IrElement.OptionalAttr, ir.indexed_attrs[1][0])
        self.assertEqual("attr2", ir.indexed_attrs[1][1])
        self.assertEqual(ge.attr.Float(1.000000), ir.indexed_attrs[1][2])

        self.assertEqual(IrElement.RequireOutput, ir.indexed_outputs[0][0])
        self.assertEqual("y1", ir.indexed_outputs[0][1])
        self.assertEqual("DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8", ir.indexed_outputs[0][2])

        self.assertEqual(IrElement.DynamicOutput, ir.indexed_outputs[1][0])
        self.assertEqual("y2", ir.indexed_outputs[1][1])
        self.assertEqual("DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8", ir.indexed_outputs[1][2])


    def test_ge_op_basic(self):
        ir = IrDef("TestV1") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        with GeGraph():
            data1 = raw_ops.Data(
                index=0, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data1')
            data2 = raw_ops.Data(
                index=1, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data2')
            data3 = raw_ops.Data(
                index=2, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data3')

            tensor = ge_op(op_type='TestV1', inputs={'x1': data1, 'x2': data2, 'x3': data3},
                           attrs={"attr1": ge.attr.Int(1), "attr2": ge.attr.Float(1.000000)},
                           outputs=["y1", ("y2", 2)],
                           ir=ir)

            self.assertTrue(isinstance(tensor[0], raw_ops.Tensor))
            op_def: OpDef = tensor[0].node

            self.assertEqual(op_def.type, 'TestV1')
            self.assertEqual(len(op_def.input), 3)
            self.assertEqual(op_def.input[0], 'data1:0')
            self.assertEqual(op_def.input[1], 'data2:0')
            self.assertEqual(op_def.input[2], 'data3:0')
            self.assertEqual(op_def.input_desc[0].name, 'x1')
            self.assertEqual(op_def.input_desc[1].name, 'x2')
            self.assertEqual(op_def.input_desc[2].name, 'x3')
            self.assertIn("attr1", op_def.attr)
            self.assertEqual(op_def.attr["attr1"].i, 1)
            self.assertNotIn("attr2", op_def.attr)  # 默认值被删除了
            self.assertEqual(len(op_def.output_desc), 3)
            self.assertEqual(op_def.output_desc[0].name, 'y1')
            self.assertEqual(op_def.output_desc[1].name, 'y20')
            self.assertEqual(op_def.output_desc[2].name, 'y21')

    def test_ge_op_optional(self):
        ir = IrDef("TestV1") \
            .input("x1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_input("x2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .optional_input("x3", "DT_FLOAT16, DT_BF16") \
            .required_attr("attr1", ge.attr.Int) \
            .attr("attr2", ge.attr.Float(1.000000)) \
            .output("y1", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8") \
            .dynamic_output("y2", "DT_FLOAT16, DT_BF16, DT_FLOAT32, DT_INT8")
        with GeGraph():
            data1 = raw_ops.Data(
                index=0, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data1')
            data2 = raw_ops.Data(
                index=1, dtype=ge.DataType.DT_FLOAT, placement='CPU', node_name='data2')

            tensor = ge_op(op_type='TestV1', inputs={'x1': data1, 'x2': data2, 'x3': None},
                           attrs={"attr1": ge.attr.Int(
                               1), "attr2": ge.attr.Float(3.000000), },
                           outputs=["y1", ("y2", 2)],
                           ir=ir)

            self.assertTrue(isinstance(tensor[0], raw_ops.Tensor))
            op_def: OpDef = tensor[0].node
            self.assertEqual(op_def.type, 'TestV1')
            self.assertEqual(len(op_def.input), 2)  # 可选输入不占位
            self.assertIn("attr1", op_def.attr)
            self.assertEqual(op_def.attr["attr1"].i, 1)
            self.assertIn("attr2", op_def.attr)  # 非默认值不能删
            self.assertEqual(op_def.attr["attr2"].f, 3.000000)

    def test_check_cann_compat(self):
        from torchair.core import _torchair
        error_msg = _torchair.check_cann_compat(
            "TestV1", ["x3"], ["attr2"])
        self.assertEqual(error_msg, "")

        error_msg = _torchair.check_cann_compat(
            "TestV1", ["inputname"], ["attr2"])
        self.assertEqual(error_msg,
                         f"optype TestV1 unsupport optional input [inputname], optional attr [], "
                         f"please upgrade cann version.")

    def test_upsample_nearest2d_decomposition_adjust_dynamic(self):
        def get_dumped_py_file_list(dir_path, file_extension='.py'):
            return [i for i in os.listdir(dir_path) if i.startswith('dynamo_') and i.endswith(f'{file_extension}')]

        for file_name in get_dumped_py_file_list('./'):
            os.remove(os.path.join('./', file_name))
        config = CompilerConfig()
        config.debug.graph_dump.type = "py"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x2):
                y2 = torch.nn.functional.interpolate(
                    x2, scale_factor=2, mode="nearest")
                return y2

        model2 = Model()
        model_dynamic = torch.compile(
            model2, backend=npu_backend, dynamic=True)
        unused = model_dynamic(torch.randn(2, 2, 2, 2))

        dumped_py_file_list = get_dumped_py_file_list('./')
        dumped_py_file_list.sort(
            key=lambda file_name: os.path.getmtime(os.path.join('./', file_name)))
        assert dumped_py_file_list.__len__() > 0
        file_name = os.path.join('./', dumped_py_file_list[-1])

        with open(file_name, 'r') as f:
            src = f.read()
        self.assertTrue("torch.ops.aten.upsample_nearest2d.default" in src)


if __name__ == '__main__':
    unittest.main()
