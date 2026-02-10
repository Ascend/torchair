import logging
import os
import unittest
from unittest.mock import Mock, patch
import shutil

import torch
import torch_npu
import torchair
from torch._subclasses.fake_tensor import FakeTensorMode
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

torch._logging.set_logs(dynamo=logging.INFO)
torch.manual_seed(7)
torch.npu.manual_seed_all(7)
logger.setLevel(logging.DEBUG)


def find_op(gm, op_default):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == op_default:
            return True

    return False


def create_optimize_wrapper(assert_func):
    original_func = torchair.npu_fx_compiler._optimize_fx

    def wrapper(gm, config, observer):
        ret = original_func(gm, config, observer)
        assert_func(gm)
        return ret

    return wrapper


class AclgraphTest(unittest.TestCase):

    def setUp(self) -> None:
        self.optimize_fx_bak = torchair.npu_fx_compiler._optimize_fx
        if not hasattr(torch.npu, "fake_record_stream"):
            patch_dynamo()        
        return super().setUp()

    def tearDown(self) -> None:
        if self.optimize_fx_bak is not None:
            torchair.npu_fx_compiler._optimize_fx = self.optimize_fx_bak
        return super().tearDown()
        
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

    def test_aclgraph_static_kernel_debug_time(self):
        class Model1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)

            def forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

        x = torch.randn([2,2], dtype=torch.float16).npu()
        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        npu_mode1 = Model1().npu()
        npu_backend1 = torchair.get_npu_backend(compiler_config=config)
        npu_mode1 = torch.compile(npu_mode1, fullgraph=True, backend=npu_backend1, dynamic=True)

        with self.assertLogs(logger, level="DEBUG") as cm1:
            _ = npu_mode1(x)
        self.assertTrue(
            any("The operation of [static kernel] execute dump json phase" in log for log in cm1.output),
            f"Expected DEBUG 'The operation of [static kernel] execute dump json phase' "
            f"not found in logs: {cm1.output}"
        )
        self.assertTrue(
            any("The operation of [static kernel] static compile phase" in log for log in cm1.output),
            f"Expected DEBUG 'The operation of [static kernel] static compile phase' "
            f"not found in logs: {cm1.output}"
        )
        self.assertTrue(
            any("The operation of [static kernel] install static kernel run pkgs phase" in log for log in cm1.output),
            f"Expected DEBUG 'The operation of [static kernel] install static kernel run pkgs phase' "
            f"not found in logs: {cm1.output}"
        )

        x = torch.randn([2, 2], dtype=torch.float32).npu()
        with self.assertLogs(logger, level="INFO") as cm2:
            _ = npu_mode1(x)
        self.assertFalse(
            any("The operation of [static kernel]" in log for log in cm2.output),
            f"Not Expected DEBUG 'The operation of [static kernel]' "
            f"found in logs: {cm2.output}"
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

    def assert_addrmsnorm_quant(self, after_gm, expect_fused=True):
        """
        Check whether the pattern fusion of add_rms_norm + quantize is successful.
        """
        check_rules = [
        (torch.ops.npu.npu_add_rms_norm_quant.default, expect_fused),
        (torch.ops.npu.npu_add_rms_norm.default, not expect_fused),
        (torch.ops.npu.npu_quantize.default, not expect_fused),
        ]

        for torch_op, expect_exist in check_rules:
            found = find_op(after_gm, torch_op)
            if expect_exist:
                self.assertTrue(found, f"Expected operator '{torch_op}' but not find")
            else:
                self.assertFalse(found, f"Not expected operator '{torch_op}' but find")

    def get_quant_input(self, last_axis, dtype1, dtype2, dtype3):
        """
        Get the input of the add_rms_norm + quantize pattern.
        """
        x1 = torch.randn(1, 2, last_axis, dtype=dtype1, device='npu')
        x2 = torch.randn(1, 2, last_axis, dtype=dtype1, device='npu')
        gamma = torch.ones(last_axis, dtype=dtype1, device='npu')
        scales = torch.ones(last_axis, dtype=dtype2, device='npu')
        zero_points = torch.zeros(last_axis, dtype=dtype3, device='npu')
        return x1, x2, gamma, scales, zero_points

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_addrmsnorm_quant(self):

        def f(x1, x2, gamma, scales, zero_points, div_mode=True):
            x1 = x1.reshape([1, -1, 16])
            x2 = x2.reshape([1, -1, 16])
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma, 4e-6)
            yOut = torch_npu.npu_quantize(y, scales, zero_points, torch.qint8, axis=-1, div_mode=div_mode)
            return yOut, xOut
        
        def f_static(x1, x2, gamma, scales, zero_points):
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma, 1e-6)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.int8, axis=-1)
            return yOut, xOut

        torchair.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_addrmsnorm_quant(gm, True))
        compile_model = torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)

        # test divmode=True
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.float16, torch.float, torch.int32)
        y1, y2 = f(x1, x2, gamma, scales, zero_points)
        y3, y4 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y2, y4))

        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        y1, y2 = f(x1, x2, gamma, scales, zero_points)
        y3, y4 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y2, y4))

        # test divmode=False
        y1, y2 = f(x1, x2, gamma, scales, zero_points, False)
        y3, y4 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y2, y4))

        # test static
        compile_model = torch.compile(f_static, backend="npugraph_ex", fullgraph=True, dynamic=False)
        y1, y2 = f_static(x1, x2, gamma, scales, zero_points)
        y3, y4 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y2, y4))


    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_addrmsnorm_quant_mismatched(self):
    
        def f(x1, x2, gamma, scales, zero_points, out_dtype=torch.qint8, div_mode=True):
            x1 = x1.reshape([1, -1, 16])
            x2 = x2.reshape([1, -1, 16])
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
            yOut = torch_npu.npu_quantize(y, scales, zero_points, out_dtype, axis=-1, div_mode = div_mode)
            return yOut, xOut

        def f_use(x1, x2, gamma, scales, zero_points):
            x1 = x1.reshape([1, -1, 16])
            x2 = x2.reshape([1, -1, 16])
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.qint8, axis=-1)
            yOut = y + yOut
            return yOut, xOut

        def f_noreshape(x1, x2, gamma, scales, zero_points):
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.qint8, axis=-1, div_mode=True)
            return yOut, xOut

        torchair.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_addrmsnorm_quant(gm, False))
        compile_model = torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)

        # test uint8 zero_poin
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.float16, torch.float16, torch.uint8)
        f(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)

        # test int8 zero_point
        zero_points = torch.zeros(16, dtype=torch.int8, device='npu')
        f(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)
        
        # test out_dtype=int32
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        f(x1, x2, gamma, scales, zero_points, torch.int32)
        compile_model(x1, x2, gamma, scales, zero_points, torch.int32)

        # test use value npu_add_rms_norm output
        compile_model = torch.compile(f_use, backend="npugraph_ex", fullgraph=True, dynamic=True)
        f_use(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)
        
        # test div_mode=False type mismatch
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.float16, torch.float16, torch.float16)
        compile_model = torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)
        f(x1, x2, gamma, scales, zero_points, div_mode=False)
        compile_model(x1, x2, gamma, scales, zero_points, div_mode=False)

        # # test last axis not aligned 32byte
        compile_model = torch.compile(f_noreshape, backend="npugraph_ex", fullgraph=True, dynamic=False)
        x1, x2, gamma, scales, zero_points = self.get_quant_input(3, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        f_noreshape(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)

        # test symint
        compile_model = torch.compile(f_noreshape, backend="npugraph_ex", fullgraph=True, dynamic=True)
        f_noreshape(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_addrmsnorm_quant_with_diff_stream(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.event1 = torchair.ops.npu_create_tagged_event(tag="22")
                self.event2 = torchair.ops.npu_create_tagged_event(tag="33")
    
            def forward(self, x1, x2, gamma, scales, zero_points):
                y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
                torchair.ops.npu_tagged_event_record(self.event1)
                with torchair.scope.npu_stream_switch('2', 3):
                    torchair.ops.npu_tagged_event_wait(self.event1)
                    yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.qint8, axis=-1)
                    torchair.ops.npu_tagged_event_record(self.event2)
                    torchair.ops.npu_record_tagged_stream(yOut, '2')
                torchair.ops.npu_tagged_event_wait(self.event2)
                return yOut, xOut

        torchair.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_addrmsnorm_quant(gm, False))

        model = Model()
        compile_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)

        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        compile_model(x1, x2, gamma, scales, zero_points)

    @unittest.skipIf(torch.__version__ < "2.7", "pattern_fusion_pass skip_duplicates is unsupported when torch < 2.7")
    def test_pattern_pass_addrmsnorm_quant_skip_duplicates(self):
        def f(x1, x2):
            return x1 + x2
        def search_fn(x1, x2, gamma, scales, zero_points, epsilon, dtype):
            y, _, x_out = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma, epsilon)
            y_out = torch.ops.npu.npu_quantize.default(y, scales, zero_points=zero_points, dtype=dtype, axis=-1)
            return y_out, x_out

        def replace_fn(x1, x2, gamma, scales, zero_points, epsilon, _):
            y1, _, x_out = torch.ops.npu.npu_add_rms_norm_quant.default(x1, x2, gamma, scales, zero_points, axis=-1, epsilon=epsilon)
            return y1, x_out

        fake_mode = FakeTensorMode()
        with fake_mode:
            # sizes/values don't actually matter for initial trace
            # once we get a possible match we re-trace with the actual values and verify the match still holds
            torchair.register_replacement(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
                scalar_workaround={"epsilon": 2e-6, "dtype": 1},
                skip_duplicates=True
        )
        torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)


    @unittest.skipIf(torch.__version__ < "2.6", "torch_npu stream api is unsupported when torch < 2.6")
    def test_stream_event_replace_in_fx(self):
        def cus_func(t):
            s = torch.npu.Stream()
            tmp = torch.add(t, 2)
            event = torch.npu.Event()        
            event.record()
            with torch.npu.stream(s):
                event.wait(s)
                r = torch.relu(tmp)
                r.record_stream(s)
            return r

        def my_backend(gm: torch.fx.GraphModule, example_inputs):
            from torchair._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
            gm = replace_stream_event_pass(gm)
            fx_target_list = []
            for node in gm.graph.nodes:
                if hasattr(node.target, "__name__"):
                    fx_target_list.append(node.target.__name__)
                else:
                    fx_target_list.append(node.target)   

                if node.name == "record":
                    self.assertEqual(node.args[0], "event")
                if node.name == "wait":
                    self.assertEqual(node.args[0], "event")
                if node.name == "set_stream":
                    self.assertIn("stream", node.args[1])
                

            print(f"fx_target_list is :{fx_target_list}")
            torchair_ir_list = ('tagged_event_record',
                                'tagged_event_wait',
                                'record_tagged_stream_',
                                'scope_enter',
                                'scope_exit')
            
            for torchair_ir in torchair_ir_list:
                self.assertIn(torchair_ir, fx_target_list)             
            return gm

        opt_m = torch.compile(cus_func, backend=my_backend, fullgraph=True, dynamic=False)
        i = torch.randn([3, 3]).to('npu')
        r = opt_m(i)  


    @unittest.skipIf(torch.__version__ < "2.6", "torch_npu stream api is unsupported when torch < 2.6")
    def test_stream_event_replace_with_set_stream(self):
        def cus_func(t):
            default_stream = torch.npu.current_stream()
            s1 = torch.npu.Stream()
            s2 = torch.npu.Stream()
            s3 = torch.npu.Stream()
            tmp = torch.add(t, 0)
            torch.npu.set_stream(s1)
            tmp = torch.add(tmp, 1)
            current_s1 = torch.npu.current_stream()
            torch.npu.set_stream(s2)
            tmp = torch.add(tmp, 2)
            torch.npu.set_stream(s3)
            tmp = torch.add(tmp, 3)
            torch.npu.set_stream(current_s1)
            r = torch.add(tmp, 4)
            torch.npu.set_stream(default_stream)
            return r

        def my_backend(gm: torch.fx.GraphModule, example_inputs):
            from torchair._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
            gm = replace_stream_event_pass(gm)
            print(f'after replace graph is : {gm.graph}')
            fx_target_list = []
            fx_node_name_list = []
            for node in gm.graph.nodes:
                fx_node_name_list.append(node.name)
                fx_target_list
                if hasattr(node.target, "__name__"):
                    fx_target_list.append(node.target.__name__)
                else:
                    fx_target_list.append(node.target)   

            self.assertEqual(fx_target_list.count('scope_enter'), fx_target_list.count('scope_exit'))
            self.assertEqual(fx_node_name_list[13], 'set_stream_3')
            self.assertEqual(fx_target_list[13], 'set_stream')
            self.assertEqual(fx_target_list[14], 'scope_exit')
            self.assertEqual(fx_node_name_list[14], 'scope_exit_1')
            self.assertEqual(fx_target_list[15], 'scope_exit')
            self.assertEqual(fx_node_name_list[15], 'scope_exit')
            self.assertEqual(fx_node_name_list[-2], 'scope_exit_2')
            self.assertEqual(fx_target_list[-2], 'scope_exit')

            return gm

        opt_m = torch.compile(cus_func, backend=my_backend, fullgraph=True, dynamic=False)
        i = torch.randn([3, 3]).to('npu')
        r = opt_m(i) 


    @unittest.skipIf(torch.__version__ < "2.6", "torch_npu stream api is unsupported when torch < 2.6")
    def test_stream_event_replace_without_set_default(self):
        def cus_func(t):
            s1 = torch.npu.Stream()
            tmp = torch.add(t, 0)
            torch.npu.set_stream(s1)
            r = torch.add(tmp, 1)
            return r

        def my_backend(gm: torch.fx.GraphModule, example_inputs):
            from torchair._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
            gm = replace_stream_event_pass(gm)
            return gm

        opt_m = torch.compile(cus_func, backend=my_backend, fullgraph=True, dynamic=False)
        i = torch.randn([3, 3]).to('npu')
        with self.assertRaises(RuntimeError) as context:
            r = opt_m(i)   
        self.assertIn("When use npugraph_ex, you must make sure at the end of your code set stream to the same stream "
                        "as the begin of your code", str(context.exception))                           


def patch_dynamo():
    from torch._dynamo.variables.user_defined import UserDefinedClassVariable

    def patch_user_defined_class_variable():
        import functools
        original_method = UserDefinedClassVariable._in_graph_classes
        
        @staticmethod
        @functools.lru_cache(None)
        def patched_in_graph_classes():
            result = original_method()
            result.add(torch.npu.Event)  
            result.add(torch.npu.Stream) 
            return result
        UserDefinedClassVariable._in_graph_classes = patched_in_graph_classes


    def fake_record_stream(self, s):
        """
        let dynamo trace Tensor.record_stream as this emtpy function,
        and you can replace it later in your compile backend to an actual function
        """
        if isinstance(self, torch._subclasses.fake_tensor.FakeTensor):
            return
        raise RuntimeError("tensor.record_stream is not supported on torch.compile! "
                        "You should write a pass to replace torch.npu.fake_record_stream to an actual function in FX graph "
                        "before aot_autograd.")

    def patch_record_stream():
        torch.npu.fake_record_stream = fake_record_stream

        def method_record_stream(self, s):
            tx = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
            return torch._dynamo.variables.TorchInGraphFunctionVariable(
                torch.npu.fake_record_stream
            ).call_function(tx, [self, s], {})
        
        torch._dynamo.variables.tensor.TensorVariable.method_record_stream = method_record_stream

    def patch_variable_builder():
        original_warp = torch._dynamo.variables.builder.VariableBuilder._wrap
        def _patch_wrapper(self, value):
            if isinstance(value, torch.npu.Event):
                self.install_guards(torch._dynamo.guards.GuardBuilder.ID_MATCH)
                torch._dynamo.utils.store_user_object_weakref(value)
                event_proxy = self.tx.output.create_proxy(
                    "call_function",
                    torch._dynamo.utils.get_user_object_from_id,
                    (id(value),),
                    {},
                )
                torch._dynamo.utils.set_example_value(event_proxy.node, value)
                out = torch._dynamo.variables.ctx_manager.EventVariable(
                    event_proxy,
                    value,
                    source=self.source,
                )
                return out
            return original_warp(self, value)

        torch._dynamo.variables.builder.VariableBuilder._wrap = _patch_wrapper


    def patch_builtin_variable():
        origin_call_id = torch._dynamo.variables.builtin.BuiltinVariable.call_id

        def _wrap_call_id(self, tx, *args):
            if torch._dynamo.variables.builtin.istype(args[0], torch._dynamo.variables.ctx_manager.EventVariable):
                return torch._dynamo.variables.ConstantVariable.create(id(args[0].value))
            return origin_call_id(self, tx, *args)

        torch._dynamo.variables.builtin.BuiltinVariable.call_id = _wrap_call_id

    patch_user_defined_class_variable()
    patch_record_stream()
    patch_variable_builder()
    patch_builtin_variable()


if __name__ == '__main__':
    unittest.main()