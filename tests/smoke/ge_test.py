import logging
import os
import unittest
import dataclasses
from typing import List

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import torchair.inference

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

    def test_inplace_input_output_option_for_remove_tensormove(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, x3, x4, x5):
                add = torch.add(x1, 10, out=x1)
                mul = torch.mul(x2, 10, out=x2)
                res1 = torch.add(add, mul)
                res2 = x3 * 100
                res3 = x4 + 10
                res4 = x5 + 100
                return res1, res2, res3, res4

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(10, 21, dtype=torch.float16, device='npu')
        x2 = torch.randn(10, 21, dtype=torch.float16, device='npu')
        x3 = torch.randn(10, 23, dtype=torch.float16, device='npu')
        x4 = torch.randn(10, 24, dtype=torch.float16, device='npu')
        x5 = torch.randn(10, 25, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output1, compile_output2, compile_output3, compile_output4 = model_compile(x1, x2, x3, x4, x5)

        self.assertTrue(
            any("ge.exec.outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )

    def test_inplace_input_output_option_for_remove_tensormove1(self):
        bs = 16
        num_head = 4
        k_head_size = 32
        v_head_size = 64
        num_blocks = 2
        lastDim_k = 16
        block_size = 32
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, key, value, slot_mapping, key_cache, value_cache):
                torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        import numpy as np
        key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
        value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
        key_cache = np.random.randn(num_blocks, num_head*k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
        value_cache = np.zeros((num_blocks, num_head*v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
        slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

        key_npu = torch.from_numpy(key).npu()
        value_npu = torch.from_numpy(value).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(key_npu, value_npu, slot_mapping_npu, key_cache_npu, value_cache_npu)

        self.assertTrue(
            any("ge.exec.outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )

    def test_inplace_input_output_option_for_cache_compile(self):
        config = CompilerConfig()

        @dataclasses.dataclass
        class InputMeta:
            data: torch.Tensor
            is_prompt: bool

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)
                self.cached_decode = torchair.inference.cache_compile(self.decode, config=config)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def _forward(self, x, kv):
                return self.linear2(x.data) + self.linear2(kv[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        x = InputMeta(data=torch.randn(2, 2).npu(), is_prompt=True)
        kv = [torch.randn(2, 2).npu()]
        model = Model().npu()
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            res_prompt = model(x, kv)
        self.assertFalse(
            any("ge.exec.outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )
        x.is_prompt = False
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            res_decode = model(x, kv)
        self.assertFalse(
            any("ge.exec.outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )

if __name__ == '__main__':
    unittest.main()