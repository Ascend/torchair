import unittest

import npu_extension_for_inductor
import torch


class InductorNpuBackendTest(unittest.TestCase):
    def test_abs(self):
        @torch.compile(dynamic=True)
        def test_abs(x):
            return torch.abs(x)

        x = torch.full((16, 512), -1.0, dtype=torch.float16)
        y = test_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x)))

    def test_abs_sqrt(self):
        @torch.compile(dynamic=True)
        def test_abs_sqrt(x):
            x = torch.abs(x)
            x = torch.sqrt(x)
            return x

        x = torch.full((32, 8, 16), -1.0, dtype=torch.float16)
        y = test_abs_sqrt(x)

        self.assertTrue(torch.allclose(y, torch.sqrt(torch.abs(x))))

    def test_softmax(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
        y = test_softmax(x)

        self.assertTrue(torch.allclose(y, torch.softmax(x, dim=3)))

    def test_constant(self):
        @torch.compile(dynamic=False)
        def test_constant(x):
            x = torch.add(x, 2.0)
            return x

        x = torch.ones(2, dtype=torch.float16)
        y = test_constant(x)

        self.assertTrue(torch.allclose(y, torch.add(x, 3.0)))

    def test_embeding(self):
        @torch.compile(dynamic=True)
        def test_embeding(x, w):
            x = torch.nn.functional.embedding(x, w)
            return x

        x = torch.ones(2, dtype=torch.int64)
        w = torch.arange(0, 200, dtype=torch.float16).view(10, 20)
        y = test_embeding(x, w)

        self.assertTrue(torch.allclose(y, torch.nn.functional.embedding(x, w)))


if __name__ == '__main__':
    unittest.main()
