import logging
import unittest

import torchair
import torchair.ge as ge
from torchair.ge_concrete_graph import ge_apis as raw_ops
from torchair.ge_concrete_graph.ge_graph import GeGraph
from torchair.ge_concrete_graph.ge_ir_pb2 import OpDef

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
        self.assertEqual(len(op_def.input), 0)
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


if __name__ == '__main__':
    unittest.main()