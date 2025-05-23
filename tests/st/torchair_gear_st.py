import unittest
import os
import contextlib
import logging
import torch
os.environ['TNG_LOG_LEVEL'] = '0'
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair.inference import set_dim_gears
from torchair.core._backend import TorchNpuGraph
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
from torchair.ge._ge_graph import GeGraph, DataType
from torchair.core._backend import initialize_graph_engine
from torchair._ge_concrete_graph.fx2ge_converter import ExecutorType, Placement, _normalize_ge_graph
from torchair.inference._gear_utils import generate_dynamic_dims_option
from torchair.ge._ge_graph import _ValueType, _GeInputInfo


logger.setLevel(logging.DEBUG)


class PatchAttr:
    def __init__(self, obj, attr_name, new_value):
        self.obj = obj
        self.attr_name = attr_name
        self.new_value = new_value
        self.original_value = None

    def __enter__(self):
        if hasattr(self.obj, self.attr_name):
            self.original_value = getattr(self.obj, self.attr_name)
            setattr(self.obj, self.attr_name, self.new_value)
        else:
            raise AttributeError(f"{self.obj} does not have attribute {self.attr_name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.obj, self.attr_name, self.original_value)


def raise_exception(*args, **kwargs):
    raise Exception("Should not be called")


@contextlib.contextmanager
def forbidden_attr(obj, attr_name):
    with PatchAttr(obj, attr_name, raise_exception):
        yield


def set_graph_output_dtypes(graph, dtypes):
    _normalize_ge_graph(graph)
    graph.attr["_output_dtypes"].list.i.extend(dtypes)
    graph.attr["_executor_type"].i = ExecutorType.NPU
    input_placements = dict()
    for op in graph.op:
        if op.type == "Data":
            input_placements[op.attr['index'].i] = Placement.HOST if op.output_desc[
                                                                         0].device_type == "CPU" else Placement.DEVICE

    for _, v in sorted(input_placements.items()):
        graph.attr["_input_placements"].list.i.append(v)


@contextlib.contextmanager
def set_env_var(key, value):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


class TorchairSt(unittest.TestCase):
    def test_set_dim_gears_twice(self):
        x = torch.ones([2, 2])
        set_dim_gears(x, {1: [1, 2]})
        set_dim_gears(x, {1: [1, 2]}) # not raise error
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0: [1, 2], 1: [1, 2]})
        exception = cm.exception
        self.assertEqual(str(exception), f"Tensor {x} already has set dim gears, "
                         f"and it is not supported to set it again.")

    def test_set_dim_gears_dim_index_not_it(self):
        x = torch.ones([2, 2])
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0.3: [1, 2]})
        exception = cm.exception
        self.assertEqual(str(exception), "Dim index in dim_gears must be an integer, but got <class 'float'>.")

    def test_set_dim_gears_value_not_list_or_tuple(self):
        x = torch.ones([2, 2])
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0: 1})
        exception = cm.exception
        self.assertEqual(str(exception), "Gears for dim index 0 in dim_gears must be a list or tuple, "
                         "but got <class 'int'>.")

    def test_set_dim_gears_dim_index_out_of_range(self):
        x = torch.ones([2, 2])
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0: [1, 2], 2: [1, 2]})
        exception = cm.exception
        self.assertEqual(str(exception), "Dim index in dim_gears must be in range [0, 1], but got 2.")

    def test_set_dim_gears_value_not_int(self):
        x = torch.ones([2, 2])
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0: [1, 1.5]})
        exception = cm.exception
        self.assertEqual(str(exception), "Element at index 1 of value for dim index 0 in dim_gears must "
                         "be an integer, but got <class 'float'>.")

    def test_set_dim_gears_value_lens_out_of_range(self):
        x = torch.ones([2, 2])
        with self.assertRaises(AssertionError) as cm:
            set_dim_gears(x, {0: list(range(1, 102))})
        exception = cm.exception
        self.assertEqual(str(exception), "Length of gears for dim index 0 in dim_gears must be in range [2, 100],"
                         " but got 101.")

    def test_not_parse_stack_over(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * 2
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        y = torch.ones([8, 8, 8])
        # 校验不会触发guard的parse stack over
        set_dim_gears(y, {0: list(range(1, 99))})
        model(y)

    def test_guard_gears(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x * 2
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        x = torch.ones([8, 8])
        set_dim_gears(x, {0: [8, 4]})
        model(x)
        y = torch.ones([4, 8])
        with forbidden_attr(GeConcreteGraph, '_normalize_ge_option'):
            model(y)  # hit compile graph

    def test_config_product(self):
        input_info = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, -1], dim_gears={0: [8, 4], 1: [8, 12]}, device_type="CPU")
        named_inputs_info = {'arg2_1': input_info}
        result = generate_dynamic_dims_option(named_inputs_info, "product")
        self.assertTrue(result['ge.inputShape'] == 'arg2_1:-1,-1')
        self.assertTrue('8,8' in result['ge.dynamicDims'])
        self.assertTrue('8,12' in result['ge.dynamicDims'])
        self.assertTrue('4,8' in result['ge.dynamicDims'])
        self.assertTrue('4,12' in result['ge.dynamicDims'])

    def test_config_zip(self):
        input_info2 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [8, 4]}, device_type="CPU")
        named_inputs_info2 = {'arg2_1': input_info2}
        result = generate_dynamic_dims_option(named_inputs_info2, "zip")
        self.assertTrue(result['ge.inputShape'] == 'arg2_1:-1,8')
        self.assertTrue('8' in result['ge.dynamicDims'])
        self.assertTrue('4' in result['ge.dynamicDims'])

    def test_config_zip_list_lens_same(self):
        input_info3 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [8, 4]}, device_type="CPU")
        input_info4 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [8, 4, 3]}, device_type="CPU")
        named_inputs_info2 = {'arg1': input_info3, 'arg2': input_info4}
        with self.assertRaises(AssertionError) as cm:
            result = generate_dynamic_dims_option(named_inputs_info2, "zip")
        exception = cm.exception
        self.assertEqual(str(exception), 'when dynamic_gears_merge_policy is zip, input gears len must same.')

    def test_config_repeat(self):
        # 验证带重复档位的设置也是支持的
        input_info6 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, -1], dim_gears={0: [1, 2, 4, 1, 2, 4], 1: [10, 10, 10, 20, 20, 20]}, device_type="CPU")
        named_inputs_info = {'arg2_1': input_info6}
        result = generate_dynamic_dims_option(named_inputs_info, "zip")

        self.assertTrue(result['ge.inputShape'] == 'arg2_1:-1,-1')
        self.assertTrue('1,10' in result['ge.dynamicDims'])
        self.assertTrue('2,10' in result['ge.dynamicDims'])
        self.assertTrue('4,10' in result['ge.dynamicDims'])
        self.assertTrue('1,20' in result['ge.dynamicDims'])
        self.assertTrue('2,20' in result['ge.dynamicDims'])
        self.assertTrue('4,20' in result['ge.dynamicDims'])

    def test_gears_num_not_over_100(self):
        input_info5 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [1, 2, 3, 4, 5, 6, 7, 8, 9]}, device_type="CPU")
        input_info6 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}, device_type="CPU")
        input_info7 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]}, device_type="CPU")
        named_inputs_info4 = {'arg1': input_info5,
                              'arg2': input_info6, 'arg3': input_info7}

        with self.assertRaises(AssertionError) as cm:
            generate_dynamic_dims_option(named_inputs_info4, "product")
        exception = cm.exception
        self.assertEqual(str(exception), "The total number of gears set cannot exceed 100, "
                         "and the current number of gears is: 900")

    def test_gears_deduplicated(self):
        # test 档位去重功能
        input_info5 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [1, 2]}, device_type="CPU")
        input_info6 = _GeInputInfo(
            value_type=_ValueType.TENSOR,
            func=None,
            shape=[-1, 8], dim_gears={0: [10, 11, 10, 11]}, device_type="CPU")
        named_inputs_info4 = {'arg1': input_info5, 'arg2': input_info6}
        result = generate_dynamic_dims_option(named_inputs_info4, "product")
        self.assertEqual(result["ge.dynamicDims"], "1,11;2,10;2,11;1,10")

    def test_muti_gear_npu_executor_pre_assigned_outputs(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph:
            x = ge.Data(index=0, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            z = ge.Mul(x, y)
            z.set_meta(torch.ones([2, 2], dtype=torch.int32, device='npu'))            
            output = ge.NetOutput([z])

            set_graph_output_dtypes(graph, [DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"ge.dynamicDims": "8;12",
                                          "ge.dynamicNodeType": "1",
                                          "ge.inputShape": "arg:-1,2"})
            executor.set_hint_shape([[-1, 2], []], [[-1, 2]])
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            y = torch.ones([], dtype=torch.int32)
            z = executor.run([x, y], [x])
            z = executor.run([x, y], [x])
            self.assertTrue(z[0] is x)

    def test_muti_gear_npu_executor_muti_output(self):
        initialize_graph_engine()
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        _privateuse1_backend.register_hook()
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        with GeGraph() as graph, set_env_var("ST_GEARS_STUB_OUTPUTSHAPE", "output_use_input_shape"):
            x = ge.Data(index=0, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            y = ge.Data(index=1, shape=[],
                        dtype=DataType.DT_INT32, placement='CPU')
            w = ge.Data(index=2, shape=[2, 2],
                        dtype=DataType.DT_INT32, placement='NPU')
            z1 = ge.Mul(x, y)
            z1.set_meta(torch.ones([2, 2], dtype=torch.int32, device='npu'))            
            z2 = ge.Mul(w, y)
            z2.set_meta(torch.ones([2, 2], dtype=torch.int32, device='npu'))            
            output = ge.NetOutput([z1, z2])

            set_graph_output_dtypes(graph, [DataType.DT_INT32, DataType.DT_INT32])

            executor = TorchNpuGraph()
            executor.load(graph, options={"ge.dynamicDims": "2;4",
                                          "ge.dynamicNodeType": "1",
                                          "ge.inputShape": "arg:-1,2;;2,2",
                                          "frozenInput": "0,1,1"})
            executor.set_hint_shape([[-1, 2], [], [2, 2]], [[-1, 2], [2, 2]])
            executor.compile()

            x = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            y = torch.ones([], dtype=torch.int32)
            w = torch.ones([2, 2], dtype=torch.int32, device='npu').to(npu_device)
            z1, z2 = executor.run([x, y, w])
            self.assertEqual(z1.shape, torch.Size([2, 2]))
            self.assertEqual(z2.shape, torch.Size([2, 2]))
            z1, z2 = executor.run([x, y, w])
            self.assertEqual(z1.shape, torch.Size([2, 2]))
            self.assertEqual(z2.shape, torch.Size([2, 2]))
            x_new = torch.ones([4, 2], dtype=torch.int32, device='npu').to(npu_device)
            y_new = torch.ones([], dtype=torch.int32)
            w_new = torch.ones([4, 2], dtype=torch.int32, device='npu').to(npu_device)
            z1_new, z2_new = executor.run([x_new, y_new, w_new])
            self.assertEqual(z1_new.shape, torch.Size([4, 2]))
            self.assertEqual(z2_new.shape, torch.Size([4, 2]))


if __name__ == '__main__':
    unittest.main()
