import logging
import os
import unittest

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

torch._logging.set_logs(dynamo=logging.INFO)
torch.manual_seed(7)
torch.npu.manual_seed_all(7)
logger.setLevel(logging.DEBUG)


class GeTest(unittest.TestCase):

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_batch_matmul_transpose_for_ge(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_ge(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.pattern_fusion_pass = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_ge_KN(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 511, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 511, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_ge_view(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(0, 1)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.remove_noop_ops = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_ge_view1(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(0, 2)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.remove_noop_ops = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_transpose_batchmatmul_convertor_default_perm_y(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, perm_y, batch_split_factor=1):
                out1 = torch_npu.npu_transpose_batchmatmul(x1, x2, batch_split_factor=batch_split_factor)
                out2 = torch_npu.npu_transpose_batchmatmul(x1, x2, perm_y=perm_y, batch_split_factor=batch_split_factor)

                return out1, out2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.remove_noop_ops = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(4, 16, 128, dtype=torch.float16, device='npu')
        x2 = torch.randn(4, 128, 128, dtype=torch.float16, device='npu')

        compile_output1, compile_output2 = model_compile(x1, x2, (1, 0, 2))
        self.assertTrue(torch.allclose(compile_output1, compile_output2))

if __name__ == '__main__':
    unittest.main()