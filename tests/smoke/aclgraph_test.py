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


if __name__ == '__main__':
    unittest.main()