import dataclasses
import copy
import logging
import os
import unittest
from typing import List
from pathlib import Path
import io
import sys
from unittest.mock import patch

import torch
import torch_npu

from torch._subclasses.fake_tensor import FakeTensorMode

import npugraph_ex
from npugraph_ex._acl_concrete_graph import replace_stream_event
from npugraph_ex.configs.compiler_config import CompilerConfig
from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
from npugraph_ex.core.utils import logger
from npugraph_ex._acl_concrete_graph.static_kernel import static_compile
from npugraph_ex.configs.npugraphex_config import _NpuGraphExConfig

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
    original_func = npugraph_ex.npu_fx_compiler._optimize_fx

    def wrapper(gm, config, observer):
        ret = original_func(gm, config, observer)
        assert_func(gm)
        return ret

    return wrapper


class AclgraphTest(unittest.TestCase):

    def setUp(self) -> None:
        self.optimize_fx_bak = npugraph_ex.npu_fx_compiler._optimize_fx
        from npugraph_ex._acl_concrete_graph import cat_optimization
        self.optimize_cat_bak = cat_optimization.optimize_cat_with_out_tensor
        if not hasattr(torch.npu, "fake_record_stream"):
            patch_dynamo()
        replace_stream_event.GraphCounter.set_graph_id(-1)
        return super().setUp()

    def tearDown(self) -> None:
        if self.optimize_fx_bak is not None:
            npugraph_ex.npu_fx_compiler._optimize_fx = self.optimize_fx_bak
        if self.optimize_cat_bak is not None:
            from npugraph_ex._acl_concrete_graph import cat_optimization
            cat_optimization.optimize_cat_with_out_tensor = self.optimize_cat_bak
        return super().tearDown()

    def test_a_aclgraph_memory_state_setting(self):
        def test_func(x):
            return x + x

        test_func = torch.compile(test_func, backend="npugraph_ex", dynamic=True)

        a = torch.randn(2, 3, device="npu:1")
        res = test_func(a)
        self.assertEqual(res.device, torch.device("npu:1"))

        b = torch.randn(4, 3, device="npu:1")
        res = test_func(b)
        self.assertEqual(res.device, torch.device("npu:1"))

    def assert_pattern_pass(self, graph_after, check_exist):
        fusion_cast_op_found_after = False
        fusion_dq_op_found_after = False

        for node in graph_after.graph.nodes:
            if node.op == "call_function":
                if node.target == torch.ops.npu.npu_add_rms_norm_dynamic_quant.default:
                    fusion_dq_op_found_after = True
                if node.target == torch.ops.npu.npu_add_rms_norm_cast.default:
                    fusion_cast_op_found_after = True

        if check_exist:
            self.assertTrue(
                fusion_cast_op_found_after,
                "npu_add_rms_norm_cast should exist in the graph after fusion"
            )
            self.assertTrue(
                fusion_dq_op_found_after,
                "npu_add_rms_norm_dynamic_quant should exist in the graph after fusion"
            )
        else:
            self.assertFalse(
                fusion_cast_op_found_after,
                "npu_add_rms_norm_cast should not exist in the graph after fusion"
            )
            self.assertFalse(
                fusion_dq_op_found_after,
                "npu_add_rms_norm_dynamic_quant should not exist in the graph after fusion"
            )

    def test_aclgraph_cache_with_static_kernel(self):
        class CachedAclGraphModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = torch.npu.npugraph_ex.inference.cache_compile(self.prompt, options=options)

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

        static_compile_call_count = 0
        def wrapped_static_compile(func):
            def wrapper(*args, **kwargs):
                nonlocal static_compile_call_count
                static_compile_call_count += 1
                return func(*args, **kwargs)
            return wrapper

        static_compile_bak = npugraph_ex._acl_concrete_graph.static_kernel.static_compile
        npugraph_ex._acl_concrete_graph.static_kernel.static_compile = wrapped_static_compile(static_compile_bak)

        options = {"static_kernel_compile": True, "inplace_pass": False, "input_inplace_pass": False}
        mm = CachedAclGraphModel()

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
        from npugraph_ex.inference._cache_compiler import CompiledModel, ModelCacheSaver
        from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        static_kernel_config = npugraph_ex.CompilerConfig()
        _process_kwargs_options(static_kernel_config, {"options":options})
        prompt_cache_bin = CompiledModel.get_cache_bin(mm.prompt, config=static_kernel_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        graph_res1 = mmc(query_prefill, query, key, value, scale, lengthq, length, length2, narrow_start)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        prompt_cache_dir_path = Path(prompt_cache_dir)
        outputs_dirs = [d for d in prompt_cache_dir_path.iterdir() if d.is_dir() and d.name == "static_kernel_compile_outputs"]
        self.assertEqual(len(outputs_dirs), 1)
        ts_outputs_dirs = [d for d in outputs_dirs[0].iterdir() if
                        d.is_dir() and d.name.endswith("_outputs") and d.name.startswith("ts")]
        self.assertEqual(len(ts_outputs_dirs), 1)
        run_pkgs = list(ts_outputs_dirs[0].glob("*.run"))
        self.assertTrue(len(run_pkgs) >= 1)
        self.assertTrue(static_compile_call_count, 1)

        mm2 = CachedAclGraphModel().npu()
        with self.assertLogs(logger, level="DEBUG") as cm:
            graph_res2 = mm2(query_prefill, query, key, value, scale, lengthq, length, length2, narrow_start)
        self.assertTrue(
            any("Rebasing" in log for log in cm.output),
            f"Expected DEBUG cache_compile 'Rebasing'"
            f"not found in logs: {cm.output}"
        )
        self.assertFalse(
            any("static kernel run eager success" in log for log in cm.output),
            f"Not Expected DEBUG 'static kernel run eager success'"
            f"found in logs: {cm.output}"
        )
        self.assertTrue(static_compile_call_count, 1) # no static compile
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled


    def test_aclgraph_cache_with_static_kernel_multi_model(self):
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

                # 通过torchair.inference.cache_compile实现编译缓存
                self.cached_prompt = torch.npu.npugraph_ex.inference.cache_compile(self.prompt, options=options)
                self.cached_decode = torch.npu.npugraph_ex.inference.cache_compile(self.decode, options=options)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                # 添加调用新函数的判断逻辑
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def _forward(self, x, kv):
                return self.linear2(x.data) + self.linear2(kv[0])

            # 重新封装为prompt函数
            def prompt(self, x, y):
                return self._forward(x, y)

            # 重新封装为decode函数
            def decode(self, x, y):
                return self._forward(x, y)

        options = {"static_kernel_compile": True, "inplace_pass": False, "input_inplace_pass": False}
        from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        static_kernel_config = npugraph_ex.CompilerConfig()
        _process_kwargs_options(static_kernel_config, {"options":options})

        model = Model().npu()
        from npugraph_ex.inference._cache_compiler import CompiledModel, ModelCacheSaver
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=static_kernel_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        decode_cache_bin = CompiledModel.get_cache_bin(model.decode, config=static_kernel_config)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(decode_cache_bin)))

        x = InputMeta(data=torch.randn(2, 2).npu(), is_prompt=True)
        kv = [torch.randn(2, 2).npu()]

        res_prompt = model(x, kv)
        x.is_prompt = False
        res_decode = model(x, kv)

        pid = os.getpid()
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))
        prompt_cache_dir_path = Path(prompt_cache_dir)
        prompt_out_path = [d for d in prompt_cache_dir_path.iterdir() if d.is_dir() and d.name.endswith("_outputs")]
        self.assertEqual(len(prompt_out_path), 1)
        prompt_ts_path = [d for d in prompt_out_path[0].iterdir() if d.is_dir() and str(pid) in d.name]
        self.assertEqual(len(prompt_ts_path), 1)

        decode_cache_dir = os.path.abspath(os.path.dirname(decode_cache_bin))
        decode_cache_dir_path = Path(decode_cache_dir)
        decode_out_path = [d for d in decode_cache_dir_path.iterdir() if d.is_dir() and d.name.endswith("_outputs")]
        self.assertEqual(len(decode_out_path), 1)
        decode_ts_path = [d for d in decode_out_path[0].iterdir() if d.is_dir() and str(pid) in d.name]
        self.assertEqual(len(decode_ts_path), 1)

        first_opcompile_path = [d for d in prompt_ts_path[0].iterdir() if d.is_dir() and d.name.endswith("_opcompile")]
        second_opcompile_path = [d for d in decode_ts_path[0].iterdir() if d.is_dir() and d.name.endswith("_opcompile")]
        self.assertEqual(len(first_opcompile_path), 1)
        self.assertEqual(len(second_opcompile_path), 1)
        first_opcompile_selected_path = [d for d in prompt_ts_path[0].iterdir() if d.is_dir() and d.name.endswith("_opcompile_selected")]
        second_opcompile_selected_path = [d for d in decode_ts_path[0].iterdir() if d.is_dir() and d.name.endswith("_opcompile_selected")]
        self.assertEqual(len(first_opcompile_selected_path), 0)
        self.assertEqual(len(second_opcompile_selected_path), 0)
        first_json = [d for d in first_opcompile_path[0].iterdir() if d.is_dir() and d.name.endswith(".json")]
        second_json = [d for d in second_opcompile_path[0].iterdir() if d.is_dir() and d.name.endswith(".json")]
        self.assertEqual(len(first_json), len(second_json))
        from collections import Counter
        self.assertEqual(Counter(first_json), Counter(second_json))

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


        npugraph_ex.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_pattern_pass(gm, True))
        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        compile_res = model_compile(x1, x2, gamma, smooth_scale1)
        expected = model(x1, x2, gamma, smooth_scale1)
        self.assertEqual(len(compile_res), len(expected))
        for comp, exp in zip(compile_res, expected):
            self.assertTrue(torch.allclose(comp, exp, atol=1e-5))

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


        npugraph_ex.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_pattern_pass(gm, False))
        options = {"pattern_fusion_pass": False}
        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options=options)

        x1 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 2, 3, dtype=torch.float16, device='npu')
        gamma = torch.ones(3, dtype=torch.float16, device='npu')
        smooth_scale1 = torch.ones(3, dtype=torch.float16, device='npu')

        compile_res = model_compile(x1, x2, gamma, smooth_scale1)

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

        compiled_model1 = torch.compile(model1, backend="npugraph_ex", options={"inplace_pass": False}, fullgraph=True, dynamic=True)

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

        options = {"inplace_pass": False}

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

        compiled_model1 = torch.compile(model1, backend="npugraph_ex", options=options, fullgraph=True, dynamic=False)

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

        options = {"inplace_pass": False, "clone_input": False}

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

        compiled_model1 = torch.compile(model1, backend="npugraph_ex", options=options, fullgraph=True, dynamic=False)

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
        compiled_model = torch.compile(model, fullgraph=True, backend="npugraph_ex", dynamic=True)

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

        def parallel_abs_sub_1(gm, example_inputs, config: CompilerConfig):
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

        def parallel_abs_sub_2(gm, example_inputs, config: CompilerConfig):
            fx_graph = gm.graph
            for node in fx_graph.nodes:
                if node.op == "call_function" and node.target == torch.ops.aten._softmax.default:
                    with fx_graph.inserting_before(node):
                        fx_graph.call_function(torch.ops.air.scope_enter.default, args=(
                            ["_user_stream_label"], ["stream1"]))

                if node.op == "call_function" and node.target == torch.ops.aten.split_with_sizes.default:
                    with fx_graph.inserting_after(node):
                        fx_graph.call_function(torch.ops.air.scope_exit.default, args=())

        options = {
            "clone_input": False,
            "inplace_pass": False,
            "post_grad_custom_pre_pass": parallel_abs_sub_1,  # parallel_abs_sub将在优化原生fx图前执行
            "post_grad_custom_post_pass": parallel_abs_sub_2 # parallel_abs_sub将在优化原生fx图后执
        }

        # 以下结果为大模型推理结果
        input0 = torch.randn(12, 6, dtype=torch.float32).npu()
        input1 = torch.randn(6, 6, dtype=torch.float32).npu()
        input2 = torch.randn(12, 6, dtype=torch.float32).npu()

        npu_mode = Network()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend="npugraph_ex", options=options, dynamic=False)
        npu_mode(input0, input1, input2)

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_batch_matmul_transpose_for_aclgraph(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

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

        options = {"pattern_fusion_pass": False}
        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options=options)

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_batch_matmul_transpose_for_aclgraph_KN(self):
        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1, x2).transpose(1, 0)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

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

        options = {"remove_noop_ops": False}
        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options=options)

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

        options = {"remove_noop_ops": False}
        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options=options)

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

        def f_no_xout(x1, x2, gamma, scales, zero_points):
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.qint8, axis=-1)
            return yOut

        def f_no_xout_with_epsilon(x1, x2, gamma, scales, zero_points):
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma, 4e-6)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.int8, axis=-1)
            return yOut

        npugraph_ex.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_addrmsnorm_quant(gm, True))
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

        # test static
        compile_model = torch.compile(f_static, backend="npugraph_ex", fullgraph=True, dynamic=False)
        y1, y2 = f_static(x1, x2, gamma, scales, zero_points)
        y3, y4 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))
        self.assertTrue(torch.equal(y2, y4))

        # test no xout
        compile_model = torch.compile(f_no_xout, backend="npugraph_ex", fullgraph=True, dynamic=False)
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        y1 = f_no_xout(x1, x2, gamma, scales, zero_points)
        y3 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))

        compile_model = torch.compile(f_no_xout_with_epsilon, backend="npugraph_ex", fullgraph=True, dynamic=False)
        x1, x2, gamma, scales, zero_points = self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        y1 = f_no_xout_with_epsilon(x1, x2, gamma, scales, zero_points)
        y3 = compile_model(x1, x2, gamma, scales, zero_points)
        self.assertTrue(torch.equal(y1, y3))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_addrmsnorm_quant_mismatched(self):
    
        def f(x1, x2, gamma, scales, zero_points, out_dtype=torch.qint8, div_mode=True):
            x1 = x1.reshape([1, -1, 16])
            x2 = x2.reshape([1, -1, 16])
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma)
            yOut = torch_npu.npu_quantize(y, scales, zero_points, out_dtype, axis=-1, div_mode=div_mode)
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

        def f_no_xout(x1, x2, gamma, scales, zero_points):
            y, _, xOut = torch_npu.npu_add_rms_norm(x1, x2, gamma, 4e-6)
            yOut = torch_npu.npu_quantize(y, scales, zero_points=zero_points, dtype=torch.int8, axis=-1, div_mode=False)
            return yOut

        npugraph_ex.npu_fx_compiler._optimize_fx = create_optimize_wrapper(lambda gm: self.assert_addrmsnorm_quant(gm, False))
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
        
        # test div_mode=False
        compile_model = torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)
        f(x1, x2, gamma, scales, zero_points, div_mode=False)
        compile_model(x1, x2, gamma, scales, zero_points, div_mode=False)

        compile_model = torch.compile(f_no_xout, backend="npugraph_ex", fullgraph=True, dynamic=False)
        f_no_xout(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)

        # test mismatch shape
        gamma = torch.ones(1, 2, 16, dtype=torch.bfloat16, device='npu')
        scales = torch.ones(16, dtype=torch.bfloat16, device='npu')
        zero_points = torch.zeros(16, dtype=torch.bfloat16, device='npu')
        f(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)
        
        scales = torch.ones(1, dtype=torch.bfloat16, device='npu')
        zero_points = torch.zeros(1, dtype=torch.bfloat16, device='npu')
        f(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)
        
        # test last axis not aligned 32byte
        compile_model = torch.compile(f_noreshape, backend="npugraph_ex", fullgraph=True, dynamic=False)
        x1, x2, gamma, scales, zero_points = self.get_quant_input(3, torch.bfloat16, torch.bfloat16, torch.bfloat16)
        f_noreshape(x1, x2, gamma, scales, zero_points)
        compile_model(x1, x2, gamma, scales, zero_points)

        # test symint
        compile_model = torch.compile(f_noreshape, backend="npugraph_ex", fullgraph=True, dynamic=True)
        f_noreshape(x1, x2, gamma, scales, zero_points)
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
            torch.npu.npugraph_ex.register_replacement(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=self.get_quant_input(16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
                scalar_workaround={"epsilon": 2e-6, "dtype": 1},
                skip_duplicates=True
        )
        torch.compile(f, backend="npugraph_ex", fullgraph=True, dynamic=True)

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph(self):
        """Test transpose+bmm+transpose pattern fusion (perm_x1=[1,0,2])."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

        # x1 (M, B, K) -> transpose(0,1) -> (B, M, K)，与 x2 (B, K, N) batch 对齐
        x1 = torch.randn(4, 64, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph(self):
        """Test transpose+bmm+transpose with pattern_fusion_pass disabled."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options={"pattern_fusion_pass": False})

        x1 = torch.randn(4, 64, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph_KN(self):
        """Test transpose+bmm+transpose when K/N not aligned (no fusion)."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

        x1 = torch.randn(4, 64, 511, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 511, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph_view(self):
        """Test transpose+bmm+transpose with transpose(1, 0) on output (no-op view case)."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(1, 0)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options={"remove_noop_ops": False})

        x1 = torch.randn(4, 64, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph_view1(self):
        """Test transpose+bmm+transpose with transpose(0, 2) on output (unsupported dims, no fusion)."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 2)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex", options={"remove_noop_ops": False})

        x1 = torch.randn(4, 64, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph_view2(self):
        """Test matmul(x1.transpose(0,2), x2).transpose(0, 1); input transpose(0,2) unsupported (dims>1), no fusion."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 2), x2).transpose(0, 1)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

        # x1 按 transpose(0,2) 反向：输入为 (K, M, B)，transpose(0,2) 后得 (B, M, K)，再与 x2 matmul
        x1 = torch.randn(512, 4, 64, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

    @unittest.skipIf(torch.__version__ < "2.6", "pattern_fusion_pass is unsupported when torch < 2.6")
    def test_close_pattern_pass_transpose_batch_matmul_transpose_for_aclgraph_KB_constraint(self):
        """Test perm_x1=[1,0,2] constraint K*B < 65536: when K*M >= 65536 fusion is skipped, result still correct."""

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output

        model = DsModel()
        model_compile = torch.compile(model, backend="npugraph_ex")

        # x1 (M, B, K) -> transpose(0,1) -> (B, M, K)；bmm 左输入 (B, M, K) 约束取 b=B,k=K 则 K*B
        # 要 K*B>65536 且 128 倍数：B=64, K=1152 -> K*B=73728，fusion rejected
        x1 = torch.randn(4, 64, 1152, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 1152, 128, dtype=torch.float16, device='npu')

        eager_output = model(x1, x2)
        with self.assertLogs(logger, level="DEBUG") as cm, torch.no_grad():
            compile_output = model_compile(x1, x2)

        self.assertTrue(torch.allclose(eager_output, compile_output))

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
            from npugraph_ex._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
            gm = replace_stream_event_pass(gm)
            fx_target_list = []
            for node in gm.graph.nodes:
                if hasattr(node.target, "__name__"):
                    fx_target_list.append(node.target.__name__)
                else:
                    fx_target_list.append(node.target)   

                if node.name == "record":
                    self.assertEqual(node.args[0], "graph_0_event")
                if node.name == "wait":
                    self.assertEqual(node.args[0], "graph_0_event")
                if node.name == "set_stream":
                    self.assertIn("graph_0_stream", node.args[1])
                

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
            from npugraph_ex._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
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
            from npugraph_ex._acl_concrete_graph.replace_stream_event import replace_stream_event_pass
            gm = replace_stream_event_pass(gm)
            return gm

        opt_m = torch.compile(cus_func, backend=my_backend, fullgraph=True, dynamic=False)
        i = torch.randn([3, 3]).to('npu')
        with self.assertRaises(RuntimeError) as context:
            r = opt_m(i)   
        self.assertIn("When use npugraph_ex, you must make sure at the end of your code set stream to the same stream "
                        "as the begin of your code", str(context.exception))                           


    def create_cat_optimization_pass_wrapper(self, assert_func):
        """
        Create a wrapper for optimize_cat_with_out_tensor to capture FX graphs before and after.

        Args:
            assert_func: Function that takes (graph_before, graph_after) and performs assertions.
                        graph_before and graph_after are torch.fx.GraphModule instances.

        Returns:
            A wrapper function that can be used to replace optimize_cat_with_out_tensor.
        """
        # Import and save reference to original function at wrapper creation time
        from npugraph_ex._acl_concrete_graph.cat_optimization import optimize_cat_with_out_tensor
        original_func = optimize_cat_with_out_tensor

        def wrapper(gm, config=None):
            # Save graph before optimization
            graph_before = copy.deepcopy(gm)

            # Call original function
            ret = original_func(gm, config)

            # Save graph after optimization
            graph_after = copy.deepcopy(gm)

            # Call assertion function
            assert_func(graph_before, graph_after)

            return ret

        return wrapper

    def assert_cat_optimization_success(self, graph_before, graph_after):
        """Verify that cat was replaced with empty + slice + out operations."""
        cat_found_after = False
        empty_found_after = False
        slice_found_after = False
        out_ops_found_after = False

        # Check graph after optimization
        for node in graph_after.graph.nodes:
            if node.op == "call_function":
                if node.target == torch.ops.aten.cat.default:
                    cat_found_after = True
                if node.target == torch.ops.aten.empty.memory_format:
                    empty_found_after = True
                if node.target == torch.ops.aten.slice.Tensor:
                    slice_found_after = True
                # Check for operations with 'out' in kwargs
                if node.kwargs and 'out' in node.kwargs:
                    out_ops_found_after = True

        # Cat optimization should succeed: cat removed, empty+slice+out ops added
        self.assertFalse(
            cat_found_after,
            "cat node should be removed after optimization"
        )
        self.assertTrue(
            empty_found_after,
            "empty.memory_format node should be created after optimization"
        )
        self.assertTrue(
            slice_found_after,
            "slice.Tensor node should be created after optimization"
        )
        self.assertTrue(
            out_ops_found_after,
            "operations with .out parameter should be created after optimization"
        )

    def assert_cat_stream_consistency(self, graph_before, graph_after):
        """
        Verify stream label consistency after cat optimization:
        1. Pre-allocated empty tensor is on the same stream as original cat
        2. Each slice is on the same stream as its corresponding original input op
        3. Each .out op is on the same stream as its corresponding original input op
        """
        from npugraph_ex._utils.graph_utils import add_stream_label_to_node_meta

        # --- Derive original stream labels from graph_before ---
        add_stream_label_to_node_meta(graph_before)

        cat_node_before = None
        for node in graph_before.graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.aten.cat.default:
                cat_node_before = node
                break
        self.assertIsNotNone(cat_node_before, "Cat node not found in graph_before")

        cat_stream = cat_node_before.meta.get('stream_label')
        original_input_streams = [
            t.meta.get('stream_label') for t in cat_node_before.args[0]
        ]

        # --- Re-derive positional stream labels on graph_after ---
        add_stream_label_to_node_meta(graph_after)

        # 1) empty's stream should equal cat's stream
        empty_nodes = []
        for n in graph_after.graph.nodes:
            if (n.op == 'call_function'
                    and n.target == torch.ops.aten.empty.memory_format):
                empty_nodes.append(n)
        self.assertGreater(len(empty_nodes), 0, "empty node not found after optimization")
        for empty_node in empty_nodes:
            self.assertEqual(
                empty_node.meta.get('stream_label'), cat_stream,
                f"Pre-allocated empty should be on cat's stream ({cat_stream}), "
                f"got {empty_node.meta.get('stream_label')}"
            )

        # Collect optimization-created slices (first arg is the empty tensor)
        empty_set = set(empty_nodes)
        opt_slices = []
        for n in graph_after.graph.nodes:
            if (n.op == 'call_function'
                    and n.target == torch.ops.aten.slice.Tensor
                    and isinstance(n.args[0], torch.fx.Node)
                    and n.args[0] in empty_set):
                opt_slices.append(n)

        # Collect .out ops (in graph order, matching original input order)
        out_ops = [n for n in graph_after.graph.nodes
                   if n.op == 'call_function' and n.kwargs and 'out' in n.kwargs]

        self.assertEqual(
            len(out_ops), len(original_input_streams),
            f"Expected {len(original_input_streams)} out ops, got {len(out_ops)}"
        )
        self.assertEqual(
            len(opt_slices), len(original_input_streams),
            f"Expected {len(original_input_streams)} slices, got {len(opt_slices)}"
        )

        # 2) Each slice's positional stream should match original input's stream
        for i, (sl, expected) in enumerate(zip(opt_slices, original_input_streams)):
            self.assertEqual(
                sl.meta.get('stream_label'), expected,
                f"slice[{i}] should be on stream {expected}, "
                f"got {sl.meta.get('stream_label')}"
            )

        # 3) Each .out op's positional stream should match original input's stream
        for i, (op, expected) in enumerate(zip(out_ops, original_input_streams)):
            self.assertEqual(
                op.meta.get('stream_label'), expected,
                f"out_op[{i}] should be on stream {expected}, "
                f"got {op.meta.get('stream_label')}"
            )

    def test_cat_optimization_cross_stream_multiple_streams_success(self):
        """Test cat optimization with multiple streams and event synchronization.
        Also verifies stream consistency:
        - empty should be on stream2 (same as cat)
        - out_op/slice for output1 should be on default stream
        - out_op/slice for output2 should be on stream1
        - out_op/slice for output3 should be on stream2
        """
        stream1 = torch.npu.Stream()
        stream2 = torch.npu.Stream()

        def f(x, stream1, stream2):
            output1 = x.exp()
            with torch.npu.stream(stream1):
                output2 = x.exp()
            with torch.npu.stream(stream2):
                output3 = x.exp()
                result = torch.cat([output1, output2, output3], dim=0)
            return result

        x = torch.randn(8, dtype=torch.float32).npu()

        def assert_success_and_stream(graph_before, graph_after):
            self.assert_cat_optimization_success(graph_before, graph_after)
            self.assert_cat_stream_consistency(graph_before, graph_after)

        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = \
            self.create_cat_optimization_pass_wrapper(assert_success_and_stream)

        model = torch.compile(f, backend="npugraph_ex", dynamic=True)
        result = model(x, stream1, stream2)

        expected = torch.cat([x.exp(), x.exp(), x.exp()], dim=0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(True, "unsupported until cann support")
    def test_aclgraph_with_superkernel(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                torch.npu.super_kernel_scope_begin("sk1")
                z = torch.add(x, y)
                torch.npu.super_kernel_scope_end("sk1")
                return z

        model = Module()
        model_compile = torch.compile(model, backend="npugraph_ex", options={"static_kernel_compile": True, "super_kernel_optimize": True})

        x1 = torch.randn(64, 4, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')

        z = model_compile(x1, x2)
        expected = torch.add(x1, x2)
        self.assertTrue(torch.allclose(z, expected, rtol=1e-3, atol=1e-3))

    def test_aclgraph_deadlock_check(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, t):
                stream = torch.npu.Stream()
                event = torch.npu.Event()
                mul_res = torch.mul(t, 5)
                add_res = torch.add(mul_res, 2)
                event.record()
                with torch.npu.stream(stream):
                    event.wait(stream)
                    relu_res = torch.relu(add_res)
                    add_res.record_stream(stream)
                return relu_res

        model = torch.compile(Model(), backend="npugraph_ex", options={"deadlock_check": True}, dynamic=False)
        x = torch.randn([3, 3]).npu()

        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            result = model(x)
        output = captured_output.getvalue()
        self.assertTrue("No deadlock risks detected." in output)


    def test_compile_fx(self):
        """Test compile_fx with torch.compile and custom_backend."""
        from torch._functorch.aot_autograd import aot_module_simplified

        class DsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x1, x2):
                output = torch.matmul(x1.transpose(0, 1), x2).transpose(0, 1)
                return output
        
        captured_config = None
        original_get_compiler = npugraph_ex.npu_fx_compiler.get_compiler

        def wrapped_get_compiler(config):
            nonlocal captured_config
            captured_config = config
            return original_get_compiler(config)

        # 使用 compile_fx 编译图
        test_options = {
                "static_kernel_compile": False,
                "inplace_pass": False,
                "input_inplace_pass": False,
                "remove_noop_ops": False,
                "remove_cat_ops": False,
                "force_eager": False,
                "pattern_fusion_pass": False,
                "clone_input": False,
                "frozen_parameter": False,
                "reuse_graph_pool_in_same_fx": False,
                "capture_limit": 64,
                "clone_output": False,
                "dump_tensor_data": False,
                "data_dump_stage": "optimized",
                "data_dump_dir": "./"
            }

        def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
            npugraph_ex.npu_fx_compiler.get_compiler = wrapped_get_compiler
            try:
                compiled_graph = npugraph_ex.compile_fx(
                    gm,
                    example_inputs,
                    test_options
                )
            finally:
                npugraph_ex.npu_fx_compiler.get_compiler = original_get_compiler
            return compiled_graph

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            return aot_module_simplified(gm, example_inputs, fw_compiler=custom_compiler)
        
        # 准备数据
        x1 = torch.randn(4, 64, 512, dtype=torch.float16, device='npu')
        x2 = torch.randn(64, 512, 128, dtype=torch.float16, device='npu')
        model = DsModel()
        # 编译并执行
        model = torch.compile(model, backend=custom_backend, dynamic=False, fullgraph=True)
        result = model(x1, x2)
        eager_output = model(x1, x2)
        self.assertTrue(torch.allclose(result, eager_output))

        assert captured_config.experimental_config.aclgraph._aclnn_static_shape_kernel.value == '0'
        assert captured_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass.value == '1'
        assert captured_config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass.value == '1'
        assert captured_config.experimental_config.remove_noop_ops.value == '0'
        assert captured_config.debug.aclgraph.remove_cat_ops.value == '0'
        assert captured_config.debug.run_eagerly.value == '0'
        assert captured_config.experimental_config.pattern_fusion_pass.value == '0'
        assert captured_config.experimental_config.frozen_parameter.value == '0'
        assert captured_config.debug.aclgraph.disable_mempool_reuse_in_same_fx.value == '1'
        assert captured_config.debug.aclgraph.static_capture_size_limit.value == '64'
        assert captured_config.debug.aclgraph.enable_output_clone.value == '0'

        assert _NpuGraphExConfig.static_kernel_compile is False
        assert _NpuGraphExConfig.inplace_pass is False
        assert _NpuGraphExConfig.input_inplace_pass is False
        assert _NpuGraphExConfig.remove_noop_ops is False
        assert _NpuGraphExConfig.remove_cat_ops is False
        assert _NpuGraphExConfig.force_eager is False
        assert _NpuGraphExConfig.pattern_fusion_pass is False
        assert _NpuGraphExConfig.clone_input is False
        assert _NpuGraphExConfig.frozen_parameter is False
        assert _NpuGraphExConfig.reuse_graph_pool_in_same_fx is False
        assert _NpuGraphExConfig.capture_limit == 64
        assert _NpuGraphExConfig.clone_output is False
        assert _NpuGraphExConfig.dump_tensor_data is False
        assert _NpuGraphExConfig.data_dump_stage == 'optimized'
        assert _NpuGraphExConfig.data_dump_dir == './'

        options_dict = _NpuGraphExConfig.as_dict()
        assert options_dict["static_kernel_compile"] is False
        assert options_dict["inplace_pass"] is False
        assert options_dict["input_inplace_pass"] is False
        assert options_dict["remove_noop_ops"] is False
        assert options_dict["remove_cat_ops"] is False
        assert options_dict["force_eager"] is False
        assert options_dict["clone_input"] is False
        assert options_dict["frozen_parameter"] is False
        assert options_dict["reuse_graph_pool_in_same_fx"] is False
        assert options_dict["capture_limit"] == 64
        assert options_dict["clone_output"] is False
        assert options_dict["dump_tensor_data"] is False
        assert options_dict["data_dump_stage"] == 'optimized'
        assert options_dict["data_dump_dir"] == './'


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