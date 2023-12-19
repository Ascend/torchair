import os

os.environ["NPU_CORE_TYPE"] = "ai_core-ascend910B1"  # 要和stub实现、执行环境匹配
os.environ["ASCIR_NOT_READY"] = "1"  # 禁用ascir和pyautofuser，使用stub实现
import npu_extension_for_inductor
import torch
import unittest


class InductorNpuBackendTest(unittest.TestCase):
    def test_abs(self):
        """
        基础测试，测试abs算子
        """

        @torch.compile(dynamic=True)
        def test_abs(x):
            return torch.abs(x)

        x = torch.full((16, 512), -1.0, dtype=torch.float16)
        y = test_abs(x)

        self.assertTrue(torch.allclose(y, torch.abs(x)))

    def test_abs_sqrt(self):
        """
        测试不支持的算子自动fallback到单算子，当前支持abs，不支持sqrt
        """

        @torch.compile(dynamic=True)
        def test_abs_sqrt(x):
            x = torch.abs(x)
            x = torch.sqrt(x)
            return x

        x = torch.full((32, 8, 16), -1.0, dtype=torch.float16)
        y = test_abs_sqrt(x)

        self.assertTrue(torch.allclose(y, torch.sqrt(torch.abs(x))))


if __name__ == '__main__':
    unittest.main()
