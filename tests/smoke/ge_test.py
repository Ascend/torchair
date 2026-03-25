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

    def test_inplace_input_output_option_for_remove_tensormove_cpu(self):
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

        x1 = torch.randn(10, 21, dtype=torch.float16, device='cpu')
        x2 = torch.randn(10, 21, dtype=torch.float16, device='cpu')
        x3 = torch.randn(10, 23, dtype=torch.float16, device='cpu')
        x4 = torch.randn(10, 24, dtype=torch.float16, device='cpu')
        x5 = torch.randn(10, 25, dtype=torch.float16, device='cpu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output1, compile_output2, compile_output3, compile_output4 = model_compile(x1, x2, x3, x4, x5)

        self.assertTrue(
            any("ge.exec.outputReuseInputMemIndexes:" not in log and
                "Skip outputReuseInputMemIndexes" in log
                for log in cm.output),
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
        key_cache = np.random.randn(num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(
            np.float16)
        value_cache = np.zeros((num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(
            np.float16)
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

    def test_inplace_input_output_option_for_cache_compile_cpu(self):
        config = CompilerConfig()

        @dataclasses.dataclass
        class InputMeta:
            x1: torch.Tensor
            x2: torch.Tensor
            x3: torch.Tensor
            x4: torch.Tensor
            x5: torch.Tensor
            is_prompt: bool

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config)
                self.cached_decode = torchair.inference.cache_compile(self.decode, config=config)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def _forward(self, x, kv):
                add = torch.add(x.x1, 10, out=x.x1)
                mul = torch.mul(x.x2, 10, out=x.x2)
                res1 = torch.add(add, mul)
                res2 = x.x3 * 100
                res3 = x.x4 + 10
                res4 = x.x5 + 100
                return res1, res2, res3, res4

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        x = InputMeta(x1=torch.randn(2, 2),
                      x2=torch.randn(2, 2),
                      x3=torch.randn(2, 2),
                      x4=torch.randn(2, 2),
                      x5=torch.randn(2, 2),
                      is_prompt=True)
        kv = [torch.randn(2, 2)]
        model = Model().npu()
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            res_prompt = model(x, kv)
        self.assertTrue(
            any("Skip outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )
        x.is_prompt = False
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            res_decode = model(x, kv)
        self.assertTrue(
            any("Skip outputReuseInputMemIndexes:" in log for log in cm.output),
            f"not found in logs: {cm.output}"
        )

    def test_view_optimize_without_scope(self):
        """测试无scope时view优化正常工作"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 4, 8)
                t1 = v1.transpose(1, 2)
                v2 = t1.reshape(2, 32)
                result = v2 + 1
                return result

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_with_scope_boundary(self):
        """测试scope边界处view反推正确触发"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 4, 8)
                t1 = v1.transpose(1, 2)

                with torchair.scope.super_kernel("test_scope", ""):
                    v2 = t1.reshape(2, 32)
                    result = v2 + 1
                return result

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_scope_exit_boundary(self):
        """测试退出scope时view反推正确触发"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                with torchair.scope.super_kernel("test_scope", ""):
                    v1 = x.view(2, 4, 8)
                    t1 = v1.transpose(1, 2)
                v2 = t1.reshape(2, 32)
                result = v2 + 1

                return result

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_multiple_scope_regions(self):
        """测试多个scope区域的view优化独立性"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 第一个scope区域
                with torchair.scope.super_kernel("scope1", ""):
                    v1 = x.view(2, 32)
                    r1 = v1 + 1
                t1 = r1.transpose(0, 1)

                # 第二个scope区域
                with torchair.scope.super_kernel("scope2", ""):
                    v2 = t1.reshape(64)
                    r2 = v2 * 2

                return r2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_disabled_with_scope(self):
        """测试关闭view优化时scope仍正常工作"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                with torchair.scope.super_kernel("test_scope", ""):
                    v1 = x.view(2, 4, 8)
                    t1 = v1.transpose(1, 2)
                    result = t1 + 1
                return result

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_nested_scope(self):
        """测试嵌套scope场景的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)

                with torchair.scope.super_kernel("outer_scope", ""):
                    t1 = v1.transpose(0, 1)
                    # 内层嵌套scope - 进入时触发外层累积的view反推
                    with torchair.scope.super_kernel("inner_scope", ""):
                        v2 = t1.reshape(64)
                        r1 = v2 + 1
                    t2 = r1.view(8, 8)
                    r2 = t2 * 2
                return r2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_stream_scope(self):
        """测试stream类型scope的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)
                with torchair.scope.npu_stream_switch("stream_0"):
                    t1 = v1.transpose(0, 1)
                    r1 = t1 + 1
                return r1

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_mixed_scope_types(self):
        """测试不同类型scope混合使用的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)

                # super_kernel scope
                with torchair.scope.super_kernel("sk_scope", ""):
                    t1 = v1.transpose(0, 1)
                    r1 = t1 + 1
                # stream scope
                with torchair.scope.npu_stream_switch("stream_0"):
                    v2 = r1.reshape(64)
                    r2 = v2 * 2
                return r2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_nested_mixed_scope_types(self):
        """测试不同类型scope嵌套的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)

                # 外层 super_kernel scope
                with torchair.scope.super_kernel("outer_sk", ""):
                    t1 = v1.transpose(0, 1)
                    # 内层 stream scope
                    with torchair.scope.npu_stream_switch("inner_stream"):
                        v2 = t1.reshape(64)
                        r1 = v2 + 1

                    t2 = r1.view(8, 8)
                    r2 = t2 * 2

                return r2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_deep_nested_scope(self):
        """测试深层嵌套SK scope的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)
                # 第一层scope
                with torchair.scope.super_kernel("level_1", ""):
                    t1 = v1.transpose(0, 1)
                    # 第二层scope
                    with torchair.scope.super_kernel("level_2", ""):
                        v2 = t1.reshape(64)
                        # 第三层scope
                        with torchair.scope.super_kernel("level_3", ""):
                            t2 = v2.view(8, 8)
                            r1 = t2 + 1
                        r2 = r1 * 2
                    r3 = r2 - 1
                return r3

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_multiple_attr_scope(self):
        """测试带多个属性的scope的view优化"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                v1 = x.view(2, 32)

                # 带多个属性的scope
                with torchair.scope.super_kernel("sk_name", ""):
                    with torchair.scope.npu_stream_switch("stream_0"):
                        t1 = v1.transpose(0, 1)
                        r1 = t1 + 1
                    v2 = r1.reshape(64)
                    r2 = v2 * 2
                return r2

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    def test_view_optimize_scope_only_view_ops(self):
        """测试scope内只有view操作的场景"""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # scope内只有连续view操作，没有计算节点
                with torchair.scope.super_kernel("view_only_scope", ""):
                    v1 = x.view(2, 32)
                    t1 = v1.transpose(0, 1)
                    v2 = t1.reshape(64)
                result = v2 + 1

                return result

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.enable_view_optimize = True
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = Model()
        model_compile = torch.compile(model, backend=npu_backend)

        x = torch.randn(64, dtype=torch.float32, device='npu')

        eager_output = model(x)
        with torch.no_grad():
            compile_output = model_compile(x)

        self.assertTrue(torch.allclose(eager_output, compile_output))


if __name__ == '__main__':
    unittest.main()