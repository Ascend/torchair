import unittest
import torch
import torch_npu
import inductor_npu_ext

inductor_npu_ext._stub_debugging_host_only()


class TestInductorNpuExt(unittest.TestCase):

    def test_add(self):
        # Test that compilation and execution do not raise any exceptions
        try:
            @torch.compile
            def func(x, y):
                return x + y

            x = torch.randn(2)
            y = torch.randn(2)
            func(x, y)
        except Exception as e:
            self.fail(f"test_add raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
