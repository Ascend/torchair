import os
import logging
import types
import unittest
import pickle
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, patch
import torch
from packaging import version
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger, EVENT_LEVEL

logger.setLevel(logging.DEBUG)
config = CompilerConfig()
_cache_dir = "./compiled_cache"


class CacheCompileTest(unittest.TestCase):
    # test case: torch._dynamo.config.inline_inbuilt_nn_modules disabled and buffer not present in self
    def test_cache_compile_buf_not_in_self_with_inline_false(self):
        class MLAAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_cache = [[torch.tensor([]), torch.tensor([])]]
            
            def init_kv_cache(self, shape, device='npu:0'):
                if len(self.kv_cache) > 0 and len(self.kv_cache[0]) > 1:
                    for j in range(len(self.kv_cache[0])):
                        if self.kv_cache[0][j].numel() == 0 or self.kv_cache[0][j].shape != torch.Size(shape):
                            self.kv_cache[0][j] = torch.zeros(shape, device=device)
            
            def forward(self, x):
                if (len(self.kv_cache) > 0 and len(self.kv_cache[0]) > 1 and 
                    self.kv_cache[0][1].numel() > 0):
                    cache_to_use = self.kv_cache[0][1]
                else:
                    cache_to_use = torch.zeros_like(x)
                
                result = x + cache_to_use
                return result

        class SelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mla_attn = MLAAttn()
            
            def forward(self, x):
                return self.mla_attn(x)

        class DecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attention = [SelfAttention()]
            
            def forward(self, x):
                return self.self_attention[0](x)

        class Decoder(torch.nn.Module):
            def __init__(self, num_layers=1):
                super().__init__()
                self.layers = torch.nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.decoder = Decoder(num_layers=1)
                self._compiled = False

            def _move_to_npu(self):
                self.to('npu:0')
            
            def _ensure_kv_cache_initialized(self, shape):
                if hasattr(self, 'decoder') and hasattr(self.decoder, 'layers'):
                    for layer in self.decoder.layers:
                        if hasattr(layer, 'self_attention') and len(layer.self_attention) > 0:
                            attn = layer.self_attention[0]
                            if hasattr(attn, 'mla_attn'):
                                attn.mla_attn.init_kv_cache(shape, device='npu:0')

            def forward(self, x):
                forward_org_self = self
                decoder_module = forward_org_self._modules['decoder']
                layers_list = decoder_module.layers
                first_layer = layers_list[0]
                self_attention_list = first_layer.self_attention
                first_attention = self_attention_list[0]
                mla_attn_instance = first_attention.mla_attn
                kv_cache_outer_list = mla_attn_instance.kv_cache
                

                if (len(kv_cache_outer_list) > 0 and len(kv_cache_outer_list[0]) > 1 and
                    kv_cache_outer_list[0][1].numel() > 0):
                    cache_to_use = kv_cache_outer_list[0][1]
                else:
                    cache_to_use = torch.zeros_like(x)
                
                attn_output = x + cache_to_use
                decoder_output = self.decoder(attn_output)
                final_output = self.linear1(x) + self.linear2(decoder_output)
                return final_output

        class ModelRunner:
            def __init__(self, model: torch.nn.Module):
                self.model = model
                self.kv_cache_initialized = False

            def init_kvcache(self, shape=(2, 2)):
                if not self.kv_cache_initialized:
                    self.model._ensure_kv_cache_initialized(shape)
                    self.kv_cache_initialized = True

        class PipelineManager:
            def __init__(self, model: torch.nn.Module):
                self.runner = ModelRunner(model)

            def _cache_compile(self):
                torch._dynamo.config.inline_inbuilt_nn_modules = False
                
                self.runner.model._move_to_npu()
                self.runner.model._ensure_kv_cache_initialized((2, 2))
                self.cached_forward = torchair.inference.cache_compile(self.runner.model.forward, config=config,
                                                                       cache_dir=_cache_dir, dynamic=True)
                self._compiled = True

            def forward(self, x: torch.Tensor, extra_param1: Optional[torch.Tensor] = None,
                        extra_param2: Optional[List[int]] = None,):
                return self.cached_forward(x, extra_param1, extra_param2)

            def add_extra_inputs_to_forward(self):

                forward_org = self.runner.model.forward

                def forward(
                    self,
                    x: torch.Tensor,
                    extra_param1: Optional[torch.Tensor] = None,
                    extra_param2: Optional[List[int]] = None,
                ) -> torch.Tensor:
                    _ = extra_param1
                    _ = extra_param2
                    return forward_org(x)

                self.runner.model.forward = types.MethodType(forward, self.runner.model)

        # check torch version
        if version.parse(torch.__version__) < version.parse("2.6.0"):
            return

        # del cache file
        cache_path = Path(_cache_dir)
        if cache_path.exists():
            compiled_files = list(cache_path.rglob("compiled_module"))
            for file_path in compiled_files:
                if file_path.is_file():
                    file_path.unlink()

        # create model1
        model = Model()
        pipeline = PipelineManager(model)
        pipeline.runner.init_kvcache(shape=(2, 2))
        pipeline.add_extra_inputs_to_forward()
        pipeline._cache_compile()
        x = torch.randn(2, 2).npu()
        # first
        res_prompt = pipeline.forward(
            x, 
            extra_param1=torch.tensor([1, 2, 3]).npu(),
            extra_param2=[10, 20]
        )

        # second
        res_decode = pipeline.forward(
            x,
            extra_param1=torch.tensor([1, 2, 3]).npu(),
            extra_param2=[10, 20]
        )

        # create model2
        model2 = Model()
        pipeline2 = PipelineManager(model2)
        pipeline2.runner.init_kvcache(shape=(2, 2))
        pipeline2.add_extra_inputs_to_forward()
        pipeline2._cache_compile()
        # third
        with self.assertLogs(logger, level="DEBUG") as cm:
            res_decode = pipeline2.forward(
                x,
                extra_param1=torch.tensor([1, 2, 3]).npu(),
                extra_param2=[10, 20]
            )

        self.assertFalse(
            any("not supported now" in log for log in cm.output),
            f"cache rebase failed, log: {cm.output}"
        )

    def test_compiled_fx_artifacts(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                return z

        compiler_config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        model = torch.compile(Model().npu(), backend=npu_backend, dynamic=False, fullgraph=True)

        x = torch.tensor([1.0]).npu()
        y = torch.tensor([2.0]).npu()
        z = model(x, y)

        # The artifact is dumped from "npu:0"
        artifact = b'\x80\x04\x95\x8b\x07\x00\x00\x00\x00\x00\x00\x8c\x18torchair.npu_fx_compiler\x94\x8c\x14_CompiledFxArtifacts\x94\x93\x94)\x81\x94}\x94(\x8c\x07version\x94\x8c\x030.1\x94\x8c\x07py_code\x94X.\x07\x00\x00\nfrom collections import OrderedDict\nfrom typing import List, Optional, Callable, Any, Dict, Tuple, Union\nimport torch\nfrom torch.profiler import record_function\nimport torch_npu\nfrom torchair._acl_concrete_graph.acl_graph import AclGraph, AclGraphCacheInfo\nfrom torchair.configs.compiler_config import CompilerConfig\nassert_size_stride = torch._C._dynamo.guards.assert_size_stride\n\ndef forward(*args, node_info=[], is_capturing: bool = False):\n    arg0_1, arg1_1 = args\n\n    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None\n    return (add,)\n\n\ncompile_configs = {}\ncompile_configs["disable_mempool_reuse_in_same_fx"] = "0"\ncompile_configs["enable_output_clone"] = "0"\ncompile_configs["static_capture_size_limit"] = "64"\n\nacl_graph = AclGraph(fx_forward=forward, config=compile_configs)\n\naclgraph_cache_info = AclGraphCacheInfo(\n    pool=None,\n    stream=None,\n    capture_error_mode="global",\n    num_warmup_iters=0,\n    fx_graph_name="graph_2",\n    user_inputs_mapping=OrderedDict([(\'arg0_1\', 0), (\'arg1_1\', 1)]),\n    unupdated_sym_input_index=[],\n    updated_ops_param={},\n    ops_update_rulers={},\n    mutated_user_inputs=[],\n    tagged_event_names=[],\n    parameter_user_inputs=[]\n)\nacl_graph.load(aclgraph_cache_info)\n\n\n_is_first_run = True\ndef kernel(*args, **kwargs):\n\n    global _is_first_run\n    if _is_first_run:\n        _is_first_run = False\n\n        assert_size_stride(args[0], (1,), (1,))\n        assert_size_stride(args[1], (1,), (1,))\n\n    fn_key = acl_graph.compile(*args)\n\n    if acl_graph.fallback_to_eager:\n        with record_function("fx_run_eagerly"):\n            return acl_graph.fx_run_eagerly(*args, **kwargs)\n\n    acl_graph.process_input(fn_key, *args)\n\n    with record_function("acl_graph_replay"):\n        acl_graph.run(fn_key, *args)\n\n    return acl_graph.reconstruct_outputs(fn_key)\n\x94ub.'
        loaded_artifact = pickle.loads(artifact)

        loaded_model = torchair.npu_fx_compiler._CompiledFxGraph.load_artifacts(loaded_artifact)
        another_z = loaded_model(x, y)

        self.assertEqual(z.item(), another_z[0].item())


if __name__ == '__main__':
    unittest.main()