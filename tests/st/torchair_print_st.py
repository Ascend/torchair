import os
import functools
import unittest
import io
import sys
import torch
import torchair


class CapturedStdout:
    def __init__(self):
        self.stdout = io.StringIO()
        self._original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.stdout
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout

    def non_empty_lines(self):
        lines = [s.strip() for s in self.stdout.getvalue().split("\n")]
        return [s for s in lines if s]


class NpuPrintSt(unittest.TestCase):
    def test_summarize_1(self):
        def func(v):
            torchair.ops.npu_print("x =", v, summarize_size=1)
            v = torch.abs(v)
            torchair.ops.npu_print("abs(x) =", v, summarize_size=1)
            v = torch.add(v, 1)
            torchair.ops.npu_print("add(abs(x), 1) =", v, summarize_size=1)
            return v

        t = torch.ones(10, 10, dtype=torch.bfloat16)
        t = torch.neg(t)

        expect_output = """
x = [[-1 ... -1]
...
[-1 ... -1]]
abs(x) = [[1 ... 1]
...
[1 ... 1]]
add(abs(x), 1) = [[2 ... 2]
...
[2 ... 2]]
"""

        with CapturedStdout() as stdout:
            func(t)
        lines = stdout.non_empty_lines()
        expect_lines = [s.strip() for s in expect_output.split("\n") if s.strip()]
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        compiled_model = torch.compile(func, backend='aot_eager')
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_summarize_neg1(self):
        def func(v):
            torchair.ops.npu_print("x =", v, summarize_size=-1)
            v = torch.abs(v)
            torchair.ops.npu_print("abs(x) =", v, summarize_size=-1)
            v = torch.add(v, 1)
            torchair.ops.npu_print("add(abs(x), 1) =", v, summarize_size=-1)
            return v

        t = torch.ones(2, 2, dtype=torch.bfloat16)
        t = torch.neg(t)

        expect_output = """
x = [[-1 -1]
[-1 -1]]
abs(x) = [[1 1]
[1 1]]
add(abs(x), 1) = [[2 2]
[2 2]]
"""

        with CapturedStdout() as stdout:
            func(t)
        lines = stdout.non_empty_lines()
        expect_lines = [s.strip() for s in expect_output.split("\n") if s.strip()]
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        compiled_model = torch.compile(func, backend='aot_eager')
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_inplace_after_print(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1)
            v.add_(1)
            torchair.ops.npu_print(v, summarize_size=-1)
            return v

        t = torch.arange(2, dtype=torch.bfloat16)

        expect_lines = ["[0 1]", "[1 2]"]
        compiled_model = torch.compile(func, backend='aot_eager')
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_side_effect_print(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1)

        t = torch.arange(2, dtype=torch.bfloat16)

        expect_lines = ["[0 1]"]
        compiled_model = torch.compile(func, backend='aot_eager')
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_scalar_print(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1)

        compiled_model = torch.compile(func, backend='aot_eager')

        t = torch.tensor(2.5)

        expect_lines = ["2.5"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        t = torch.tensor(3)

        expect_lines = ["3"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        t = torch.tensor(True)

        expect_lines = ["1"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_bool_print(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1)

        compiled_model = torch.compile(func, backend='aot_eager')

        t = torch.tensor([True, False])

        expect_lines = ["[1 0]"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_npu_backend(self):
        def func(v):
            torchair.ops.npu_print("x =", v)
            return torch.abs(v)

        t = torch.neg(torch.ones(10, 10, dtype=torch.bfloat16))

        target_op = torch.ops.air.print.default
        converter = target_op._ge_converter

        reached = False

        @functools.wraps(converter)
        def wrapper(*args, **kwargs):
            nonlocal reached
            reached = True
            return converter(*args, **kwargs)

        target_op._ge_converter = wrapper

        compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
        compiled_model(t)

        self.assertTrue(reached)

    def test_npu_backend_with_inplace(self):
        from torchair.core import _npu_graph_executor
        import _privateuse1_backend
        npu_device = _privateuse1_backend.npu_device()
        torch.utils.rename_privateuse1_backend("npu")

        def func(v):
            torchair.ops.npu_print(v)
            v.add_(1)
            return v

        t = torch.ones((2, 2), device='npu')
        t.is_npu = True

        class FakeTorchNpu:

            def __getattr__(self, item):
                return self

            @staticmethod
            def get_npu_format(*args, **kwargs):
                return 0

        try:
            origin_torch_npu = sys.modules.get('torch_npu', None)
            sys.modules['torch_npu'] = FakeTorchNpu()
            compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
            compiled_model(t)
        finally:
            if origin_torch_npu:
                sys.modules['torch_npu'] = origin_torch_npu

        self.assertTrue(True)  # assert not raise

    def test_invalid_summarize(self):
        self.assertRaises(ValueError, torchair.ops.npu_print, "x =", torch.ones(2, 2), summarize_size=0)
        self.assertRaises(ValueError, torchair.ops.npu_print, "x =", torch.ones(2, 2), summarize_size=-2)
        self.assertRaises(TypeError, torchair.ops.npu_print, "x =", torch.ones(2, 2), summarize_size=3.2)


if __name__ == '__main__':
    unittest.main()
