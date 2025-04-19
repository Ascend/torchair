# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest
import os

import torch
import npu_extension_for_inductor

do_stub_test = os.getenv("ASCIR_NOT_READY", None) == "1"
if not do_stub_test:
    import torch_npu


class InductorNpuBackendTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.device = torch.device("npu") if not do_stub_test else torch.device("cpu")

    def check_precision(self, result, expected):
        if do_stub_test:
            self.assertFalse(torch.allclose(result, expected))
        else:
            self.assertTrue(torch.allclose(result, expected, atol=1e-3, rtol=1e-3))

    def full(self, *args, **kwargs):
        return torch.full(*args, **kwargs).to(self.device)

    def ones(self, *args, **kwargs):
        return torch.ones(*args, **kwargs).to(self.device)

    def test_abs(self):
        @torch.compile(dynamic=True)
        def test_abs(x):
            return torch.abs(x)

        x = self.full((16, 512), -1.0, dtype=torch.float16)
        y = test_abs(x)

        self.check_precision(y, torch.abs(x))

    def test_abs_sqrt(self):
        @torch.compile(dynamic=True)
        def test_abs_sqrt(x):
            x = torch.abs(x)
            x = torch.sqrt(x)
            return x

        x = self.full((32, 8, 16), -1.0, dtype=torch.float16)
        y = test_abs_sqrt(x)

        self.check_precision(y, torch.sqrt(torch.abs(x)))

    def test_constant(self):
        @torch.compile(dynamic=False)
        def test_constant(x):
            x = torch.add(x, 2.0)
            return x

        x = self.ones(2, dtype=torch.float16)
        y = test_constant(x)

        self.check_precision(y, torch.add(x, 2.0))

    def test_reduce_last_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_last_dim(x):
            x = torch.sum(x, dim=-1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_last_dim(x)

        self.check_precision(y, torch.sum(x, dim=-1))

    def test_reduce_first_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_first_dim(x):
            x = torch.sum(x, dim=0)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_first_dim(x)

        self.check_precision(y, torch.sum(x, dim=0))

    def test_reduce_middle_dim(self):
        @torch.compile(dynamic=True)
        def test_reduce_middle_dim(x):
            x = torch.sum(x, dim=1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_reduce_middle_dim(x)

        self.check_precision(y, torch.sum(x, dim=1))

    def test_slice_first_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_first_dim_then_abs(x):
            x = x[0:8]
            x = torch.abs(x)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_first_dim_then_abs(x)

        self.check_precision(y, torch.abs(x[0:8]))

    def test_slice_last_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_last_dim_then_abs(x):
            x = x[:, :, 0:8]
            x = torch.abs(x)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_dim_then_abs(x)

        self.check_precision(y, torch.abs(x[:, :, 0:8]))

    def test_slice_middle_dim_then_abs(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_dim_then_abs(x):
            x = x[:, 0:8, :]
            x = torch.abs(x)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_dim_then_abs(x)

        self.check_precision(y, torch.abs(x[:, 0:8, :]))

    def test_slice_then_reduce_last_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_last_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=-1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_last_dim(x)

        self.check_precision(y, torch.sum(x[0:8], dim=-1))

    def test_slice_then_reduce_first_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_first_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=0)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_first_dim(x)

        self.check_precision(y, torch.sum(x[0:8], dim=0))

    def test_slice_then_reduce_middle_dim(self):
        @torch.compile(dynamic=True)
        def test_slice_then_reduce_middle_dim(x):
            x = x[0:8]
            x = torch.sum(x, dim=1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_then_reduce_middle_dim(x)

        self.check_precision(y, torch.sum(x[0:8], dim=1))

    def test_slice_last_then_reduce_first(self):
        @torch.compile(dynamic=True)
        def test_slice_last_then_reduce_first(x):
            x = x[:, :, 0:8]
            x = torch.sum(x, dim=0)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_then_reduce_first(x)

        self.check_precision(y, torch.sum(x[:, :, 0:8], dim=0))

    def test_slice_last_then_reduce_middle(self):
        @torch.compile(dynamic=True)
        def test_slice_last_then_reduce_middle(x):
            x = x[:, :, 0:8]
            x = torch.sum(x, dim=1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_last_then_reduce_middle(x)

        self.check_precision(y, torch.sum(x[:, :, 0:8], dim=1))

    def test_slice_middle_then_reduce_first(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_then_reduce_first(x):
            x = x[:, 0:8, :]
            x = torch.sum(x, dim=0)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_then_reduce_first(x)

        self.check_precision(y, torch.sum(x[:, 0:8, :], dim=0))

    def test_slice_middle_then_reduce_last(self):
        @torch.compile(dynamic=True)
        def test_slice_middle_then_reduce_last(x):
            x = x[:, 0:8, :]
            x = torch.sum(x, dim=-1)
            return x

        x = self.ones(32, 64, 16, dtype=torch.float16)
        y = test_slice_middle_then_reduce_last(x)

        self.check_precision(y, torch.sum(x[:, 0:8, :], dim=-1))

    def test_broadcast_1dim(self):
        @torch.compile(dynamic=True)
        def broadcast_dim(x, y):
            return x + y

        x = self.ones(32, 64, dtype=torch.float16)
        y = self.ones(128, 3, dtype=torch.float16)[1:33, 1:2]
        z = broadcast_dim(x, y)
        self.check_precision(z, x + y)

        x = self.ones(32, 64, dtype=torch.float16)
        y = self.ones(32, 128, dtype=torch.float16)[1:2, 1:65]
        z = broadcast_dim(x, y)
        self.check_precision(z, x + y)

    def test_transpose1(self):
        @torch.compile(dynamic=True)
        def test_transpose(x, y):
            return x.t() + y

        x = self.ones(32, 64)
        y = self.ones(64, 32)
        z = test_transpose(x, y)
        self.check_precision(z, x.t() + y)

    def test_transpose2(self):
        @torch.compile(dynamic=True)
        def test_transpose(x1, x2, y):
            return x1.t() + x2.t() + y

        x1 = self.ones(32, 64)
        x2 = self.ones(32, 64)
        y = self.ones(64, 32)
        z = test_transpose(x1, x2, y)
        self.check_precision(z, x1.t() + x2.t() + y)

    def test_softmax0(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        dtype = torch.float16 if not do_stub_test else torch.float32  # Use float32 for CPU testing
        x = self.ones(32, 96, 64, 128, dtype=dtype)
        y = test_softmax(x)

        self.check_precision(y, torch.softmax(x, dim=0))

    def test_softmax1(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        dtype = torch.float16 if not do_stub_test else torch.float32  # Use float32 for CPU testing
        x = self.ones(32, 96, 64, 128, dtype=dtype)
        y = test_softmax(x)

        self.check_precision(y, torch.softmax(x, dim=1))

    def test_softmax2(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        dtype = torch.float16 if not do_stub_test else torch.float32  # Use float32 for CPU testing
        x = self.ones(32, 96, 64, 128, dtype=dtype)
        y = test_softmax(x)

        self.check_precision(y, torch.softmax(x, dim=2))

    def test_softmax3(self):
        @torch.compile(dynamic=True)
        def test_softmax(x):
            x = torch.softmax(x, dim=3)
            return x

        dtype = torch.float16 if not do_stub_test else torch.float32  # Use float32 for CPU testing
        x = self.ones(32, 96, 64, 128, dtype=dtype)
        y = test_softmax(x)

        self.check_precision(y, torch.softmax(x, dim=3))


if __name__ == '__main__':
    unittest.main()
