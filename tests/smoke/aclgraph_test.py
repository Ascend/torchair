import logging
import os
import unittest
from unittest.mock import Mock, patch
import shutil

import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

torch._logging.set_logs(dynamo=logging.INFO)
torch.manual_seed(7)
torch.npu.manual_seed_all(7)
logger.setLevel(logging.DEBUG)


class AclgraphTest(unittest.TestCase):

    def test_aclgraph_cache_with_static_kernel(self):
        class CachedAclGraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=static_kernel_config)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                return self.cached_prompt(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2,
                                          narrow_start)

            def prompt(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                return self._forward(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start)

            def _forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k = k * 1.1
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                 scale=scale, softmax_lse_flag=False,
                                                                 actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32) # narrow_start
                res = add3 * mmm.mean()
                return res


        static_kernel_config = CompilerConfig()
        static_kernel_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        static_kernel_config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass = True
        static_kernel_config.mode = "reduce-overhead"
        static_kernel_config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        static_kernel_config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./static_kernel"
        mm = CachedAclGraphModel()

        from torchair.core import _torchair
        _torchair.GetSocName()
        _torchair.AclopStartDumpArgs(1, "./static_kernel")
        _torchair.AclopStopDumpArgs(1)

        length = [28, 29, 1]
        length2 = [66, 88, 55]
        lengthq = [33, 44, 55]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()

        torch._dynamo.mark_static(query_prefill)
        torch._dynamo.mark_static(query)
        torch._dynamo.mark_static(key)
        torch._dynamo.mark_static(value)
        mmc = mm.npu()
        from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
        prompt_cache_bin = CompiledModel.get_cache_bin(mm.prompt, config=static_kernel_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        graph_res1 = mmc(query_prefill, query, key, value, scale, lengthq, length, length2, narrow_start)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        mm2 = CachedAclGraphModel().npu()
        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = mm2(query_prefill, query, key, value, scale, lengthq, length, length2, narrow_start)
        self.assertTrue(
            any("Rebasing" in log for log in cm.output),
            f"Expected DEBUG cache_compile 'Rebasing'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("static kernel run eager success" in log for log in cm.output),
            f"Expected DEBUG 'static kernel run eager success'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    def test_aclgraph_cache_recapture_with_ops_update(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=tng_config)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                return self.cached_prompt(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2,
                                          narrow_start)

            def prompt(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                return self._forward(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start)

            def _forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                 scale=scale, softmax_lse_flag=False,
                                                                 actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32) # narrow_start
                res = add3 * mmm.mean()
                return res


        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.experimental_config.keep_inference_input_mutations = True
        tng_config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True
        model1 = RecaptureModel().npu()
        length = [28, 29, 1]
        length2 = [66, 88, 55]
        lengthq = [33, 44, 55]
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)

        from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
        prompt_cache_bin = CompiledModel.get_cache_bin(model1.prompt, config=tng_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        graph_res1 = model1(query_prefill_, query_, key_, value_, scale, lengthq, length, length2, narrow_start)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled


        model2 = RecaptureModel().npu()

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = model2(query_prefill_, query_, key_, value_, scale, length_new, length2_new, lengthq_new,
                                narrow_start)
            graph_res3 = model2(query_prefill_, query_, key, value_, scale, length_new, length2_new, lengthq_new,
                                narrow_start)

        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("Record the 2 th updated node" in log for log in cm.output),
            f"Expected DEBUG 'Record the 1 th updated node'"
            f"not found in logs: {cm.output}"
        )
        self.assertFalse(
            any("Record the 3 th updated node" in log for log in cm.output),
            f"Not expected DEBUG 'Record the 2 th updated node'"
            f"found in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_for_aclgraph(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                h1 = y1.size(-1)
                y2 = y1.view(-1, h1)
                yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)

                _, _, h2 = y1.shape
                y1 = y1.view(-1, h2).to(torch.float32)

                y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
                scale1Out3_view = scale1Out3.view(-1, 1)
                return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_for_ge(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                h1 = y1.size(-1)
                y2 = y1.view(-1, h1)
                yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)

                _, _, h2 = y1.shape
                y1 = y1.view(-1, h2).to(torch.float32)

                y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
                scale1Out3_view = scale1Out3.view(-1, 1)
                return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_for_aclgraph(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                h1 = y1.size(-1)
                y2 = y1.view(-1, h1)
                yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)

                _, _, h2 = y1.shape
                y1 = y1.view(-1, h2).to(torch.float32)

                y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
                scale1Out3_view = scale1Out3.view(-1, 1)
                return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_config.experimental_config.pattern_fusion_pass = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertFalse(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_for_ge(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut, scale1Out = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                h1 = y1.size(-1)
                y2 = y1.view(-1, h1)
                yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y2, smooth_scales=smooth_scales)

                _, _, h2 = y1.shape
                y1 = y1.view(-1, h2).to(torch.float32)

                y3, _, xOut3 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                yOut3, scale1Out3 = torch_npu.npu_dynamic_quant(y3.flatten(0, 1))
                scale1Out3_view = scale1Out3.view(-1, 1)
                return yOut, xOut, scale1Out, y1, xOut1, yOut2, scale1Out2, xOut3, yOut3, scale1Out3_view


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "max-autotune"
        npu_config.experimental_config.pattern_fusion_pass = False
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertFalse(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_for_aclgraph_with_multistream(self):
        class DsModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)
                with torchair.scope.npu_stream_switch('2', 3):
                    y1, _, xOut1 = torch_npu.npu_add_rms_norm(x1, x2, weight)
                    _, _, h2 = y1.shape
                    y1 = y1.view(-1, h2).to(torch.float32)

                yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                return xOut, yOut2, scale1Out2, y1, xOut1


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel2()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertTrue(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )
        self.assertTrue(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_for_cast_with_subgraph_in_diff_stream(self):
        class DsModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)

                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([y])
                    _, _, h2 = y.shape
                    y = y.view(-1, h2).to(torch.float32)

                return y, xOut


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel2()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_cast.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_cast.default' in logs: {cm.output}"
        )

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_for_dynamicquant_with_subgraph_in_diff_stream(self):
        class DsModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2, weight, smooth_scales):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, weight)

                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.wait([y])
                    yOut2, scale1Out2 = torch_npu.npu_dynamic_quant(y, smooth_scales=smooth_scales)

                return yOut2, xOut, scale1Out2


        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel2()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            model_compile(x1, x2, gamma, smooth_scale1)

        self.assertFalse(
            any("target: npu.npu_add_rms_norm_dynamic_quant.default" in log for log in cm.output),
            f"Expected no DEBUG log 'target: npu.npu_add_rms_norm_dynamic_quant.default' in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_parameter_and_mutated(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32)
                add3 = add3 @ self.linear(x)
                res = add3 * mmm.mean()
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.experimental_config.keep_inference_input_mutations = True
        npu_backend = torchair.get_npu_backend(compiler_config=tng_config)

        model1 = RecaptureModel().npu()
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        compiled_model1 = torch.compile(model1, backend=npu_backend, fullgraph=True, dynamic=True)

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res1 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 4" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 4'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            # recapture
            graph_res3 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 7" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 7'"
            f"not found in logs: {cm.output}"
        )
        del graph_res1
        del graph_res2
        del graph_res3
        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = b
            # recapture
            graph_res4 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertFalse(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Not Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 4" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 4'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_parameter_and_mutated_clone_input_false(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32)
                add3 = add3 @ self.linear(x)
                res = add3 * mmm.mean()
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.debug.aclgraph.clone_input = False
        tng_config.experimental_config.keep_inference_input_mutations = True
        npu_backend = torchair.get_npu_backend(compiler_config=tng_config)

        model1 = RecaptureModel().npu()
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        compiled_model1 = torch.compile(model1, backend=npu_backend, fullgraph=True, dynamic=True)

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res1 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 3" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 3'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            # recapture
            graph_res3 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 6" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 6'"
            f"not found in logs: {cm.output}"
        )
        del graph_res1
        del graph_res2
        del graph_res3
        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = b
            # recapture
            graph_res4 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertFalse(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Not Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 3" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 3'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_parameter_and_mutated_dynamic_false(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32)
                add3 = add3 @ self.linear(x)
                res = add3 * mmm.mean()
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.experimental_config.keep_inference_input_mutations = True
        npu_backend = torchair.get_npu_backend(compiler_config=tng_config)

        model1 = RecaptureModel().npu()
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        compiled_model1 = torch.compile(model1, backend=npu_backend, fullgraph=True, dynamic=False)

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res1 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 4" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 4'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            # recapture
            graph_res3 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 7" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 7'"
            f"not found in logs: {cm.output}"
        )
        del graph_res1
        del graph_res2
        del graph_res3
        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = b
            # recapture
            graph_res4 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertFalse(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Not Expected DEBUG 'After capturing fx_graph'"
            f"found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 4" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 4'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_parameter_and_mutated_clone_input_false_static(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                add3 = ifa3[0]
                add3 = torch.narrow(add3, -1, 32, 32)
                add3 = add3 @ self.linear(x)
                res = add3 * mmm.mean()
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.debug.aclgraph.clone_input = False
        tng_config.experimental_config.keep_inference_input_mutations = True
        npu_backend = torchair.get_npu_backend(compiler_config=tng_config)

        model1 = RecaptureModel().npu()
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        compiled_model1 = torch.compile(model1, backend=npu_backend, fullgraph=True, dynamic=False)

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res1 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 3" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 3'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            # recapture
            graph_res3 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 6" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 6'"
            f"not found in logs: {cm.output}"
        )
        del graph_res1
        del graph_res2
        del graph_res3
        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = b
            # recapture
            graph_res4 = compiled_model1(query_prefill_, query_, key, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x)
        self.assertFalse(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Not Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 3" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 3'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_multiple_fx(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                        is_prompt=True):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                if is_prompt:
                    add3 = ifa3[0]
                    add3 = torch.narrow(add3, -1, 32, 32)
                    add3 = add3 @ self.linear(x)
                    res = add3 * mmm.mean()
                else:
                    res = ifa3[0]
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.experimental_config.keep_inference_input_mutations = True
        npu_backend = torchair.get_npu_backend(compiler_config=tng_config)

        model1 = RecaptureModel().npu()
        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        compiled_model1 = torch.compile(model1, backend=npu_backend, fullgraph=True, dynamic=True)

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res1 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x, True)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 4" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 4'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res2 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x, False)
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("'activate_num': 2" in log for log in cm.output),
            f"Expected DEBUG ''activate_num': 2'"
            f"not found in logs: {cm.output}"
        )

        with self.assertLogs(logger, level="DEBUG") as cm:
            compiled_model1.linear.weight.data = a
            graph_res3 = compiled_model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                         lengthq_new, narrow_start, x, True)
        self.assertTrue(
            any("The current AclGraph no needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph no needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_static_kernel(self):
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        class Model2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        x = torch.randn(2,2).npu()
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True

        npu_mode1 = Model1().npu()
        npu_backend1 = torchair.get_npu_backend(compiler_config=config)
        npu_mode1 = torch.compile(npu_mode1, fullgraph=True, backend=npu_backend1, dynamic=False)
        with self.assertLogs(logger, level="DEBUG") as cm1:
            _ = npu_mode1(x)
        self.assertTrue(
            any("reselect_static_kernel executed successfully" in log for log in cm1.output),
            f"Expected DEBUG 'reselect_static_kernel executed successfully' "
            f"not found in logs: {cm1.output}"
        )

        npu_mode2 = Model2().npu()
        npu_backend2 = torchair.get_npu_backend(compiler_config=config)
        npu_mode2 = torch.compile(npu_mode2, fullgraph=True, backend=npu_backend2, dynamic=False)
        with self.assertLogs(logger, level="DEBUG") as cm2:
            _ = npu_mode2(x)
        self.assertTrue(
            any(
                "Static compilation skipped" in log
                or "Using debug directory" in log
                for log in cm2.output
            ),
            f"Expected DEBUG 'Static compilation skipped' or 'Using debug directory' "
            f"not found in logs: {cm2.output}"
        )

    def test_aclgraph_userinput_construct_in_share_memory_with_cache_compile(self):
        class RecaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)
                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=tng_config)
                self.cached_decode = torchair.inference.cache_compile(self.decode, config=tng_config)

            def forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                        is_prompt=True):
                if is_prompt:
                    return self.cached_prompt(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                                              is_prompt)
                else:
                    return self.cached_decode(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                                              is_prompt)

            def prompt(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                       is_prompt):
                return self._forward(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                                     is_prompt)

            def decode(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                       is_prompt):
                return self._forward(qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                                     is_prompt)

            def _forward(self, qp, q, k, v, scale, actual_seq_lenq, actual_seq_len, actual_seq_len2, narrow_start, x,
                        is_prompt=True):
                pfa0, _ = torch_npu.npu_fused_infer_attention_score(qp, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths=actual_seq_lenq,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                q = q * scale
                ifa1, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=actual_seq_len)
                mm1 = ifa1.view([ifa1.shape[-1], -1]).clone()
                q = q + 0.01
                ifa2, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                    scale=scale, softmax_lse_flag=False,
                                                                    actual_seq_lengths_kv=[66, 166, 266])
                mm2 = ifa2.view([-1, ifa2.shape[-1]]).clone()
                mmm = torch.mm(mm1, mm2) + pfa0.mean()
                k.mul_(1.1)
                v = v / 1.1
                ifa3 = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=32, input_layout="BNSD",
                                                                scale=scale, softmax_lse_flag=False,
                                                                actual_seq_lengths_kv=actual_seq_len2)
                if is_prompt:
                    add3 = ifa3[0]
                    add3 = torch.narrow(add3, -1, 32, 32)
                    add3 = add3 @ self.linear(x)
                    res = add3 * mmm.mean()
                else:
                    res = ifa3[0]
                return res

        tng_config = torchair.CompilerConfig()
        tng_config.mode = 'reduce-overhead'
        tng_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        tng_config.experimental_config.keep_inference_input_mutations = True

        length_new = [88, 99, 1]
        length2_new = [40, 50, 60]
        lengthq_new = [99, 50, 10]
        scale = 1 / 0.0078125
        narrow_start = 32
        query_prefill_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        query_ = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value_ = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        key = key_.clone()
        torch._dynamo.mark_static(query_prefill_)
        torch._dynamo.mark_static(query_)
        torch._dynamo.mark_static(key_)
        torch._dynamo.mark_static(value_)
        x = torch.randn([32, 32]).npu()
        a = torch.ones(32, 32).npu()
        b = torch.zeros(32, 32).npu()

        model1 = RecaptureModel().npu()
        model1.linear.weight.data = a

        from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
        prompt_cache_bin = CompiledModel.get_cache_bin(model1.prompt, config=tng_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))
        decode_cache_bin = CompiledModel.get_cache_bin(model1.decode, config=tng_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(decode_cache_bin)))
        decode_cache_dir = os.path.abspath(os.path.dirname(decode_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        graph_res1 = model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                            lengthq_new, narrow_start, x, True)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        self.assertFalse(os.path.exists(decode_cache_dir))
        graph_res2 = model1(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                            lengthq_new, narrow_start, x, False)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled

        torch._dynamo.reset()
        model2 = RecaptureModel().npu()
        with self.assertLogs(logger, level="DEBUG") as cm:
            model2.linear.weight.data = a
            graph_res3 = model2(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                lengthq_new, narrow_start, x, True)
            graph_res4 = model2(query_prefill_, query_, key_, value_, scale, length_new, length2_new,
                                lengthq_new, narrow_start, x, False)
        self.assertTrue(
            any("Rebasing" in log for log in cm.output),
            f"Expected DEBUG cache_compile 'Rebasing'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("No find captured AclGraph" in log for log in cm.output),
            f"Expected DEBUG 'No find captured AclGraph'"
            f"not found in logs: {cm.output}"
        )
        self.assertTrue(
            any("After capturing fx_graph" in log for log in cm.output),
            f"Expected DEBUG 'After capturing fx_graph'"
            f"not found in logs: {cm.output}"
        )

    def test_replay_update_stream_same(self):
        class MM(torch.nn.Module):
            def forward(self, q, k, v, scale, actual_seq_len):
                ifa, _ = torch_npu.npu_fused_infer_attention_score(
                    q, k, v, num_heads=32,
                    input_layout="BNSD", scale=scale, softmax_lse_flag=False,
                    actual_seq_lengths_kv=actual_seq_len
                )
                return ifa

        length = [28, 29, 1]
        scale = 1 / 0.0078125
        query = torch.randn(3, 32, 1, 128, dtype=torch.float16).npu()
        key = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()
        value = torch.randn(3, 32, 512, 128, dtype=torch.float16).npu()

        torch._dynamo.mark_static(query)
        torch._dynamo.mark_static(key)
        torch._dynamo.mark_static(value)

        mm = MM()
        compiler_config = torchair.CompilerConfig()
        compiler_config.mode = 'reduce-overhead'
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        
        mmc = torch.compile(mm, backend=npu_backend, dynamic=True)

        replay_stream = torch.npu.Stream(priority=-1)
        print(f"replay stream: {replay_stream.stream_id}")
        with torch.npu.stream(replay_stream):
            _ = mmc(query, key, value, scale, length)
        torch.npu.synchronize()

        update_stream = torchair._acl_concrete_graph.acl_graph.CapturedGraphUpdateAndReplay._update_stream
        with self.assertLogs(logger, level="INFO") as cm:
            with torch.npu.stream(update_stream):
                _ = mmc(query, key, value, scale, length)
            torch.npu.synchronize()
        self.assertTrue(
            any("Update the stream for parameter" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured'"
            f"not found in logs: {cm.output}"
        )

    @unittest.skipIf('ATB_HOME_PATH' not in os.environ, 
                    "_npu_paged_attention is unsupported without ATB_HOME_PATH environment variable")
    def test_aclgraph_update_param_with__npu_paged_attention(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, query, key_cache, value_cache, block_table, context_lens):
                output = torch.zeros_like(query[:, :, :96])
                torch_npu._npu_paged_attention(
                    query=query, 
                    key_cache=key_cache, 
                    value_cache=value_cache,
                    num_kv_heads=16,
                    num_heads=32, 
                    scale_value=0.38888,
                    block_table=block_table,
                    context_lens=context_lens,
                    out=output,
                )
                return output + 1

        from torch._dynamo import allow_in_graph
        allow_in_graph(torch_npu._npu_paged_attention)
        model = Model()

        config = CompilerConfig()
        config.mode = 'reduce-overhead'
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        compiled_model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)

        num_blocks = 64
        num_tokens = 2
        block_size = 128
        kv_heads = 16
        head_size = 288
        num_heads = 32
        head_size_v = 96

        import random
        import numpy as np
        query_np = np.random.uniform(-1, 1, (num_tokens, num_heads, head_size)).astype(np.float16)
        key_cache_np = np.random.uniform(-1, 1, (num_blocks, block_size, kv_heads, head_size)).astype(np.float16)
        value_cache_np = np.random.uniform(-1, 1, (num_blocks, block_size, kv_heads, head_size_v)).astype(np.float16)
        max_blocks_per_seq = (1024 + block_size - 1) // block_size
        block_table_np = np.array([
            [random.randint(0, num_blocks - 1) for _ in range(max_blocks_per_seq)]
            for _ in range(num_tokens)
        ], dtype=np.int32)
        context_lens_np = np.full(num_tokens, 128, dtype=np.int32)
        context_lens_np_new = np.full(num_tokens, 512, dtype=np.int32)

        query = torch.from_numpy(query_np).npu()
        key_cache = torch.from_numpy(key_cache_np).npu()
        value_cache = torch.from_numpy(value_cache_np).npu()
        block_table = torch.from_numpy(block_table_np).npu()
        context_lens = torch.from_numpy(context_lens_np)
        context_lens_new = torch.from_numpy(context_lens_np_new)        

        torch._dynamo.mark_static(query)
        torch._dynamo.mark_static(key_cache)
        torch._dynamo.mark_static(value_cache)
        torch._dynamo.mark_static(block_table)

        eager_res1 = model(query, key_cache, value_cache, block_table, context_lens)
        eager_res2 = model(query, key_cache, value_cache, block_table, context_lens_new)

        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res1 = compiled_model(query, key_cache, value_cache, block_table, context_lens)
            self.assertTrue(torch.allclose(eager_res1, graph_res1))

            graph_res2 = compiled_model(query, key_cache, value_cache, block_table, context_lens_new)
            self.assertTrue(torch.allclose(eager_res2, graph_res2))

        self.assertTrue(
            any("Replay AclGraph and update input params successfully" in log for log in cm.output),
            f"Expected DEBUG 'Replay AclGraph and update input params successfully'"
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_scope_with_post_pass(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x, y, z):
                sqrt_01 = torch.sqrt(x)
                softmax_01 = torch.softmax(sqrt_01, dim=-1)
                abs_01 = torch.abs(softmax_01)
                split_01, split_02 = torch.split(abs_01, split_size_or_sections=[6, 6], dim=0)
                matmul_01 = torch.matmul(split_01, y)
                add_01 = torch.add(split_02, matmul_01)
                concat_01 = torch.cat([add_01, z], dim=0)
                relu_01 = self.relu(concat_01)
                transpose_01 = torch.transpose(relu_01, 0, 1)
                return transpose_01

        def parallel_abs_sub_1(gm, example_inputs, config: torchair.CompilerConfig):
            fx_graph = gm.graph
            for node in fx_graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.aten.sqrt.default:
                    with fx_graph.inserting_before(node):
                        fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                            ["_user_stream_label"], ["stream0"]))

                if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                    with fx_graph.inserting_after(node):
                        fx_graph.call_function(
                            torch.ops.air.scope_exit.default, args=())

        def parallel_abs_sub_2(gm, example_inputs, config: torchair.CompilerConfig):
            fx_graph = gm.graph
            for node in fx_graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.aten._softmax.default:
                    with fx_graph.inserting_before(node):
                        fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                            ["_user_stream_label"], ["stream1"]))

                if node.op == "call_function" and node.target == torch.ops.aten.split_with_sizes.default:
                    with fx_graph.inserting_after(node):
                        fx_graph.call_function(torch.ops.air.scope_exit.default, args=())

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.aclgraph.clone_input = False
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        config.post_grad_custom_pre_pass = parallel_abs_sub_1  # parallel_abs_sub将在torchair优化原生fx图前执行
        config.post_grad_custom_post_pass = parallel_abs_sub_2 # parallel_abs_sub将在torchair优化原生fx图后执
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        # 以下结果为大模型推理结果
        input0 = torch.randn(12, 6, dtype=torch.float32).npu()
        input1 = torch.randn(6, 6, dtype=torch.float32).npu()
        input2 = torch.randn(12, 6, dtype=torch.float32).npu()

        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_mode(input0, input1, input2)

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_batch_matmul_transpose_for_aclgraph(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
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
    def test_close_pattern_pass_batch_matmul_transpose_for_aclgraph(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
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
    def test_pattern_pass_batch_matmul_transpose_for_aclgraph_with_multistream(self):
        class DsModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.event1 = torchair.ops.npu_create_tagged_event(tag="66")
                self.event2 = torchair.ops.npu_create_tagged_event(tag="77")

            def forward(self, x1, x2):
                y = torch.matmul(x1, x2)
                torchair.ops.npu_tagged_event_record(self.event1)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.npu_tagged_event_wait(self.event1)
                    output = torch.transpose(y, 1, 0)
                    torchair.ops.npu_tagged_event_record(self.event2)
                    torchair.ops.npu_record_tagged_stream(output, '2')
                torchair.ops.npu_tagged_event_wait(self.event2)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=npu_config)
        model = DsModel2()
        model_compile = torch.compile(model, backend=npu_backend)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_aclgraph_KN(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
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
    def test_close_pattern_pass_batch_matmul_transpose_for_aclgraph_view(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(0, 1)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
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
    def test_close_pattern_pass_batch_matmul_transpose_for_aclgraph_view1(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(0, 2)
                return output

        npu_config = torchair.CompilerConfig()
        npu_config.mode = "reduce-overhead"
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

if __name__ == '__main__':
    unittest.main()