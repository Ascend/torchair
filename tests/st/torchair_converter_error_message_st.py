import logging

import torch
from torch.library import Library, impl
import torchair
import torchair.ge as ge
from torchair.ge._ge_graph import torch_type_to_ge_type
from torchair._utils.error_messages import ConverterErrorMsg
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import _validate_support
from torchair.ge.attr import _ATTR_TYPE_MAP

import unittest

logger.setLevel(logging.DEBUG)
npu_backend = torchair.get_npu_backend(compiler_config=(torchair.CompilerConfig()))

class TestConverterValidate(unittest.TestCase):
    def test_auto_converter_not_op_overload(self):
        def fake_op(x):
            return x
        with self.assertRaises(RuntimeError) as context:
            _validate_support(fake_op)

        self.assertTrue(
            ConverterErrorMsg.NOT_OP_OVERLOAD.format(target=str(fake_op)) in str(context.exception)
        )

    def test_auto_converter_builtin_op_aten(self):
        aten_op = torch.ops.aten.add.default
        with self.assertRaises(RuntimeError) as context:
            _validate_support(aten_op)

        self.assertTrue(
            ConverterErrorMsg.BUILTIN_OP.format(target=str(aten_op)) in str(context.exception)
        )

    def test_auto_converter_builtin_op_prim(self):
        prim_op = torch.ops.prim.device.default
        with self.assertRaises(RuntimeError) as context:
            _validate_support(prim_op)

        self.assertTrue(
            ConverterErrorMsg.BUILTIN_OP.format(target=str(prim_op)) in str(context.exception)
        )

    def test_auto_converter_has_scalar(self):
        m = Library("custom_definev6", "DEF")
        m.define("my_op_testv6(Tensor self, Scalar indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv6", "Meta")
        def my_op_testv6_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, z):
                return torch.ops.custom_definev6.my_op_testv6(x, 1, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32), torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(ConverterErrorMsg.SCALAR_INPUT.format(
            target="custom_definev6.my_op_testv6.default",
            name="indices"
        ) in str(context.exception))

    def test_auto_converter_so_load_failed(self):
        m = Library("npu", "FRAGMENT")
        m.define("so_load_faile(Tensor tensor) -> Tensor")

        @torch.library.impl(m, "so_load_faile", "Meta")
        def so_load_faile_meta(tensor):
            return tensor

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.npu.so_load_faile(x)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            from unittest.mock import patch

            # Mock _torchair.get_registered_ir_def 返回None, 模拟SO加载失败
            with patch('torchair.core._torchair.get_registered_ir_def') as mock_get_ir:
                mock_get_ir.return_value = ("None", None, None, None)
                model(torch.zeros(2, 2, dtype=torch.float32))
        self.assertTrue(ConverterErrorMsg.SO_LOAD_FAILED in str(context.exception))

    def test_auto_converter_failed_ir_unregistered(self):
        m = Library("custom_definev5", "DEF")
        m.define("my_op_testv5(Tensor self, Tensor indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv5", "Meta")
        def my_op_testv5_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev5.my_op_testv5(x, y, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.tensor([[1, 1], [1, 1]]),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(ConverterErrorMsg.GE_IR_NOT_REGISTERED.format(name="MyOpTestv5") in str(context.exception))

    def test_auto_converter_no_match_inputs(self):
        m = Library("npu", "FRAGMENT")
        m.define("my_op_no_match_inputs(Tensor self, Tensor updates, Tensor indices) -> Tensor")

        @torch.library.impl(m, "my_op_no_match_inputs", "Meta")
        def my_op_no_match_inputs_meta(self, updates, indices):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.npu.my_op_no_match_inputs(x, y, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.randn(2, 2, dtype=torch.float32),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.TENSOR_INPUTS_COUNT_MISMATCH.format(
                target="npu.my_op_no_match_inputs.default",
                ge_name="MyOpNoMatchInputs",
                tensor_inputs="['self', 'updates', 'indices']",
                ge_inputs="[('x', 'required'), ('updates', 'required')]"
            ) in str(context.exception))

    def test_auto_converter_no_match_attrs(self):
        m = Library("npu", "FRAGMENT")
        m.define("my_op_no_match_attrs(Tensor self, Tensor updates, *, int indices=2, int dim=2) -> Tensor")
        @torch.library.impl(m, "my_op_no_match_attrs", "Meta")
        def my_op_no_match_attrs_meta(self, updates, indices=2, dim=2):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.npu.my_op_no_match_attrs(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32), torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.NON_TENSOR_INPUTS_COUNT_MISMATCH.format(
                target="npu.my_op_no_match_attrs.default",
                ge_name="MyOpNoMatchAttrs",
                non_tensor_inputs="['indices', 'dim']",
                ge_attrs="[('indices', 'VT_INT')]"
            ) in str(context.exception))

    def test_auto_converter_no_match_outputs(self):
        m = Library("npu", "FRAGMENT")
        m.define("my_op_no_match_outputs(Tensor(a!) self, Tensor updates, *, int indices=2) -> Tensor")

        @torch.library.impl(m, "my_op_no_match_outputs", "Meta")
        def my_op_no_match_outputs_meta(self, updates, indices=2):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.npu.my_op_no_match_outputs(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32), torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.OUTPUTS_COUNT_MISMATCH.format(
                target="npu.my_op_no_match_outputs.default",
                ge_name="MyOpNoMatchOutputs",
                outputs_count=2,
                ge_outputs_count=1
            ) in str(context.exception))

    def test_auto_converter_inplace_count_mismatch(self):
        m = Library("custom_inplace", "DEF")
        m.define("my_op_inplace(Tensor(a!) self, Tensor updates) -> Tensor")

        @impl(m, "my_op_inplace", "Meta")
        def my_op_inplace_meta(self, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.custom_inplace.my_op_inplace(x, y)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32), torch.randn(2, 2, dtype=torch.float32))

        self.assertTrue(
            ConverterErrorMsg.INPLACE_COUNT_MISMATCH.format(
                target="custom_inplace.my_op_inplace.default",
                ge_name="MyOpInplace",
                inplace_count=1,
                ge_inplace_count=2
            ) in str(context.exception)
        )

    def test_custom_op_input_count_error(self):
        m = Library("custom_definev2", "DEF")
        m.define("my_op_testv2(Tensor self, Tensor indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv2", "Meta")
        def my_op_testv2_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev2.my_op_testv2(x, y, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.tensor([[1, 1], [1, 1]]),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.ARGS_COUNT_MISMATCH.format(
                op_type="MyOpTestv2",
                expected=4,
                actual=3
            ) in str(context.exception))

    def test_custom_op_input_required_error(self):
        m = Library("custom_definev1", "DEF")
        m.define("my_op_testv1(Tensor self, Tensor[] indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv1", "Meta")
        def my_op_testv1_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev1.my_op_testv1(x, [y], z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.tensor([[1, 1], [1, 1]]),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.INPUT_TYPE_REQUIRED.format(
                op_type="MyOpTestv1",
                name="updates",
                param_type="required"
            ) in str(context.exception))


    def test_custom_op_input_dynamic_error(self):
        m = Library("custom_definev3", "DEF")
        m.define("my_op_testv3(Tensor self, Tensor indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv3", "Meta")
        def my_op_testv3_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev3.my_op_testv3(x, y, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.randn(2, 2, dtype=torch.float32),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.INPUT_TYPE_DYNAMIC.format(
                op_type="MyOpTestv3",
                name="updates",
                param_type="dynamic"
            ) in str(context.exception))

    def test_custom_op_input_optional_error(self):
        m = Library("custom_definev7", "DEF")
        m.define("my_op_testv7(Tensor self, Tensor[] indices, Tensor updates) -> Tensor")
        @impl(m, "my_op_testv7", "Meta")
        def my_op_testv7_meta(self, indices, updates):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev7.my_op_testv7(x, y, z)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  [torch.randn(2, 2, dtype=torch.float32)],
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.INPUT_TYPE_OPTIONAL.format(
                op_type="MyOpTestv7",
                name="updates",
                param_type="optional"
            ) in str(context.exception))

    def test_custom_op_input_type_illegal(self):
        m = Library("custom_definev8", "DEF")
        m.define("my_op_testv8(Tensor x) -> Tensor")
        @impl(m, "my_op_testv8", "Meta")
        def my_op_testv8_meta(x):
            return x

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ops.custom_definev8.my_op_testv8(x)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.INPUT_TYPE_ILLEGAL.format(
                op_type="MyOpTestv8",
                name="x",
                param_type="illegal"
            ) in str(context.exception))

    def test_custom_op_attrs_type_error(self):
        m = Library("custom_definev4", "DEF")
        m.define("my_op_testv4(Tensor self, Tensor indices, Tensor updates, int use_index) -> Tensor")

        @impl(m, "my_op_testv4", "Meta")
        def my_op_testv4_meta(self, indices, updates, use_index):
            return self

        class Model(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.ops.custom_definev4.my_op_testv4(x, y, z, 1)

        model = torch.compile(Model(), backend=npu_backend, dynamic=True)
        with self.assertRaises(RuntimeError) as context:
            model(torch.zeros(2, 2, dtype=torch.float32),
                  torch.tensor([[1, 1], [1, 1]]),
                  torch.randn(2, 2, dtype=torch.float32))
        self.assertTrue(
            ConverterErrorMsg.ATTR_TYPE_ILLEGAL.format(
                op_type="MyOpTestv4",
                attr_type="VT_NAMED_ATTRS",
                name="use_indices",
                all_attr_types=_ATTR_TYPE_MAP.keys()
            ) in str(context.exception))

    def test_custom_op__torch_dtype_to_ge_dtype_unsupported(self):
        with self.assertRaises(RuntimeError) as context:
            torch_type_to_ge_type(torch.bits4x2)
        self.assertTrue(ConverterErrorMsg.TORCH_TYPE_TO_GE_TYPE_UN_SUPPORT.format(dtype="torch.bits4x2")
                        in str(context.exception))

    def test_custom_op_so_load_failed(self):
        from unittest.mock import patch
        with patch('torchair.core._torchair.get_registered_ir_def') as mock_get_ir:
            # 模拟 SO 加载失败的情况
            mock_get_ir.return_value = ("None", None, None, None)
            with self.assertRaises(RuntimeError) as context:
                ge.custom_op('TestOp', inputs=None, outputs=None)
            self.assertTrue(
                ConverterErrorMsg.SO_LOAD_FAILED in str(context.exception)
            )

    def test_custom_op_with_op_unregistered(self):
        with self.assertRaises(RuntimeError) as cm:
            ge.custom_op('MyOpTestv5', inputs=None, outputs=None)
        self.assertEqual(ConverterErrorMsg.GE_IR_NOT_REGISTERED.format(name="MyOpTestv5"), str(cm.exception))

if __name__ == "__main__":
    unittest.main()