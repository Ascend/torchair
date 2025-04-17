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

    def test_constant(self):
        @torch.compile(dynamic=False)
        def test_constant(x):
            x = torch.add(x, 2.0)
            return x

        x = torch.ones(2, dtype=torch.float16)
        y = test_constant(x)

        self.assertTrue(torch.allclose(y, torch.add(x, 2.0)))

    def test_reduce_last_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_last_dim(x):
            x = torch.sum(x, dim=-1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_last_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x, dim=-1)))

    def test_reduce_first_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_first_dim(x):
            x = torch.sum(x, dim=0)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_first_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x, dim=0)))

    def test_reduce_middle_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_middle_dim(x):
            x = torch.sum(x, dim=1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_middle_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x, dim=1)))

    def test_slice_first_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_first_dim_then_abs(x):
            x = x[0:8]
            x = torch.abs(x)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_first_dim_then_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x[0:8])))

    def test_slice_last_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_last_dim_then_abs(x):
            x = x[:, :, 0:8]
            x = torch.abs(x)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_dim_then_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x[:, :, 0:8])))

    def test_slice_middle_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_dim_then_abs(x):
            x = x[:, 0:8, :]
            x = torch.abs(x)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_dim_then_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x[:, 0:8, :])))

    def test_slice_then_reduce_last_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_last_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=-1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_last_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[0:8], dim=-1)))

    def test_slice_then_reduce_first_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_first_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=0)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_first_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[0:8], dim=0)))

    def test_slice_then_reduce_middle_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_middle_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_middle_dim(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[0:8], dim=1)))

    def test_slice_last_then_reduce_first(self):
        @torch.compile(dynamic=True)
        def test_slice_last_then_reduce_first(x):
            x = x[:, :, 0:8]
            x = torch.sum(x, dim=0)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_then_reduce_first(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[:, :, 0:8], dim=0)))

    def test_slice_last_then_reduce_middle(self):
        @torch.compile(dynamic=True)
        def test_slice_last_then_reduce_middle(x):
            x = x[:, :, 0:8]
            x = torch.sum(x, dim=1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_then_reduce_middle(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[:, :, 0:8], dim=1)))

    def test_slice_middle_then_reduce_first(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_then_reduce_first(x):
            x = x[:, 0:8, :]
            x = torch.sum(x, dim=0)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_then_reduce_first(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[:, 0:8, :], dim=0)))

    def test_slice_middle_then_reduce_last(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_then_reduce_last(x):
            x = x[:, 0:8, :]
            x = torch.sum(x, dim=-1)
            return x

        x = torch.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_then_reduce_last(x)

        self.assertTrue(torch.allclose(y, torch.sum(x[:, 0:8, :], dim=-1)))

    def test_softmax(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        x = torch.ones(1, 96, 2048, 128, dtype=torch.float16)
        y = test_softmax(x)

        self.assertTrue(torch.allclose(y, torch.softmax(x, dim=3)))


if __name__ == '__main__':
    unittest.main()
