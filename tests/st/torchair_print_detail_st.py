import os
import functools
import unittest
import io
import sys
import torch
import torchair
from torchair_st_utils import generate_faked_module

import _privateuse1_backend
_privateuse1_backend.register_hook()
npu_device = _privateuse1_backend.npu_device()
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())

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
    
    def test_scalar_print_detail(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1, tensor_detail=True)

        compiled_model = torch.compile(func, backend='aot_eager')

        t = torch.tensor(2.5)

        expect_lines = ["tensor(2.5, shape=[]), dtype=torch.float32)"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        t = torch.tensor(3)

        expect_lines = ["tensor(3, shape=[]), dtype=torch.int64)"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        t = torch.tensor(True)

        expect_lines = ["tensor(1, shape=[]), dtype=torch.bool)"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)
            

    def test_bool_print_detail(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1, tensor_detail=True)

        compiled_model = torch.compile(func, backend='aot_eager')

        t = torch.tensor([True, False])

        expect_lines = ["tensor([1 0], shape=[2]), dtype=torch.bool)"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_empty_print_detail(self):
        def func(v):
            torchair.ops.npu_print(v, summarize_size=-1, tensor_detail=True)

        compiled_model = torch.compile(func, backend='aot_eager')

        t = torch.empty((0, 3), dtype=torch.int32)

        expect_lines = ["tensor([], shape=[0, 3]), dtype=torch.int32)"]
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

    def test_npu_backend_detail(self):
        def func(v):
            torchair.ops.npu_print("x =", v, tensor_detail=True)
            return torch.abs(v)

        t = torch.neg(torch.ones(10, 10, dtype=torch.bfloat16))

        target_op = torch.ops.air.print.default
        converter = target_op._ge_converter
        reached = False
        expect_line = "x = tensor({}, shape={}), dtype=torch.bfloat16)"

        @functools.wraps(converter)
        def wrapper(*args, **kwargs):
            self.assertEqual(args[1], expect_line)
            nonlocal reached
            reached = True
            return converter(*args, **kwargs)

        target_op._ge_converter = wrapper
        compiled_model = torch.compile(func, backend=torchair.get_npu_backend())
        compiled_model(t)

        self.assertTrue(reached)

if __name__ == '__main__':
    unittest.main()
