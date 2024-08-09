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
    def test_cache_hint(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                torchair.ops.npu_print("x =", x, summarize_num=1)
                x = torch.abs(x)
                torchair.ops.npu_print("abs(x) =", x, summarize_num=1)
                x = torch.add(x, 1)
                torchair.ops.npu_print("add(abs(x), 1) =", x, summarize_num=1)
                return x

        t = torch.ones(10, 10, dtype=torch.bfloat16)
        t = torch.neg(t)

        model = Model()

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
            model(t)
        lines = stdout.non_empty_lines()
        expect_lines = [s.strip() for s in expect_output.split("\n") if s.strip()]
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)

        compiled_model = torch.compile(model)
        with CapturedStdout() as stdout:
            compiled_model(t)
        lines = stdout.non_empty_lines()
        self.assertEqual(len(lines), len(expect_lines))
        for x, y in zip(lines, expect_lines):
            self.assertEqual(x, y)


if __name__ == '__main__':
    unittest.main()
