import os
import re
import sys
from typing import List
import unittest
from unittest.mock import Mock
import logging

import torch
import torch.nn.functional as F
from torch import fx
import sympy

import npugraph_ex
from npugraph_ex.configs.compiler_config import CompilerConfig
from npugraph_ex.core.utils import logger
from npugraph_ex.inference._cache_compiler import CompiledModel, ModelCacheSaver
from npugraph_ex._acl_concrete_graph.utils import reconstruct_args_kwargs, WeakRef, LazyMessage
from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
from npugraph_ex.configs._option_base import CallableValue

torch._logging.set_logs(dynamo=logging.INFO)
logger.setLevel(logging.DEBUG)

from torchair_st_stub_aclgraph_utils import (
    StubNpu,
    patch_ops_npu_module,
    patch_torch_point_npu_module,
    patch_torch_npu_module,
    forbidden_attr,
    register_custom_ops,
)
from torchair_st_utils import capture_logger, capture_warnings, generate_faked_module, register_is_npu, create_cat_optimization_pass_wrapper

torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module('npu', generate_faked_module())

def buildCompileConfig(options):
    config = CompilerConfig()
    _process_kwargs_options(config, {"options":options})
    return config

class NpugraphExCacheSt(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.original_npu_module = None
        self.original_torch_point_npu_module = None
        self.original_torch_npu_module = None
        self.stub_module = StubNpu()
        register_custom_ops()
        register_is_npu()

    def setUp(self) -> None:
        self.original_npu_module = patch_ops_npu_module(self.stub_module)
        self.original_torch_point_npu_module = patch_torch_point_npu_module(self.stub_module)
        self.original_torch_npu_module = patch_torch_npu_module(self.stub_module)

        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        self.call_bak = AclConcreteGraph.__call__
        from npugraph_ex.inference._cache_compiler import CacheBackend
        self.cachebackend_fw_compiler = CacheBackend.fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        self.optimize_cat_with_out_tensor = cat_optimization.optimize_cat_with_out_tensor 
        return super().setUp()

    def tearDown(self) -> None:
        torch.ops.npu = self.original_npu_module
        torch.npu = self.original_torch_point_npu_module
        sys.modules['torch_npu'] = self.original_torch_npu_module
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
        AclConcreteGraph.__call__ = self.call_bak
        from npugraph_ex.inference._cache_compiler import CacheBackend
        CacheBackend.fw_compiler = self.cachebackend_fw_compiler
        from npugraph_ex._acl_concrete_graph import cat_optimization
        cat_optimization.optimize_cat_with_out_tensor = self.optimize_cat_with_out_tensor
        return super().tearDown()
    
    def test_aclgraph_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

            def prompt(self, x):
                return self._forward(x)
        
        options = {"clone_input": False, "inplace_pass": True}
        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([3, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x)  # cache hint
            model_match_cache(x)  # cache hint

    def test_aclgraph_cache_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = npugraph_ex.inference.cache_compile(self._forward, config=buildCompileConfig(options), dynamic=False)

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)


        options = {"clone_input": False, "input_inplace_pass": True}
        model = CacheModel()
        prompt_cache_dir = CompiledModel.get_cache_bin(model._forward, config=buildCompileConfig(options), dynamic=False)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]

        t12 = torch.zeros(1, 5)
        t22 = torch.zeros(2, 5)[:, 0:1]
        t32 = torch.zeros(5, 2)
        s12 = 5
        s22 = [2, 3]
        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(t1, t2, t3, s1, s2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(t12, t22, t32, s12, s22)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 5==10, stride 1==1 at dim=1", str(exception))

    def test_aclgraph_cache_dynamic_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = npugraph_ex.inference.cache_compile(self._forward, config=buildCompileConfig(options))

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)


        options = {"clone_input": False, "input_inplace_pass": True}
        model = CacheModel()
        prompt_cache_dir = CompiledModel.get_cache_bin(model._forward, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]

        t12 = torch.zeros(2, 5)
        t22 = torch.zeros(2, 5)[:, 0:1]
        t32 = torch.zeros(5, 2)
        s12 = 5
        s22 = [2, 3]
        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(t1, t2, t3, s1, s2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(t12, t22, t32, s12, s22)  # cache hint
            exception = cm.exception
            self.assertIn("expected size 2==1, stride 5==5 at dim=0", str(exception))

    def test_aclgraph_cache_capture_and_replay_keep_inference_input_mutations_true(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x):
                return self.cached_prompt(x)

            def prompt(self, x):
                return self._forward(x)

            def _forward(self, x):
                x.mul_(2)
                return x + 1

        options = {"clone_input": False, "input_inplace_pass": True}
        model = Model()
        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        x_ = torch.randn([3, 2])
        x = x_.clone()

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x_)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        # inference
        with self.assertLogs(logger, level="DEBUG") as cm:
            for _ in range(2):
                output = model(x)

        self.assertTrue(
            any("The current AclGraph needs to be recaptured" in log for log in cm.output),
            f"Expected DEBUG 'The current AclGraph needs to be recaptured' "
            f"not found in logs: {cm.output}"
        )

    def test_aclgraph_cache_closure_vars(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x, y):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                x = x + y
                y = y + float('inf')
                empty = torch.ops.aten.empty([2, 2])
                return (x, y, empty)

            def prompt(self, x, y):
                return self._forward(x, y)

        options = {"clone_input": False,"input_inplace_pass": True}
        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x, y)  # cache hint
            model_match_cache(x, y)  # cache hint

    def test_aclgraph_cache_with_nan(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x, y):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                x = x + y
                y = y + float('nan')
                empty = torch.ops.aten.empty([2, 2])
                return (x, y, empty)

            def prompt(self, x, y):
                return self._forward(x, y)

        options = {"clone_input": False,"input_inplace_pass": True}
        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x, y)  # cache hint
            model_match_cache(x, y)  # cache hint

    def test_aclgraph_static_capture_size_limit_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, input):
                return self.cached_prompt(input)

            def prompt(self, input):
                return input + input

        options = {"clone_input": False, "capture_limit": 1, "input_inplace_pass": True}
        model1 = Model()
        prompt_cache_dir = CompiledModel.get_cache_bin(model1.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        self.assertFalse(os.path.exists(prompt_cache_dir))

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        self.assertTrue("Success to capture fx_graph" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([4, 2]))
        self.assertTrue("static_capture_size_limit reached" in stdout.getvalue())

        with capture_logger() as stdout:
            model1(torch.randn([3, 2]))
        # fall back to eager no aclgraph log
        self.assertTrue("Find captured AclGraph" not in stdout.getvalue())
        self.assertTrue("Success to capture fx_graph" not in stdout.getvalue())

        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    def test_aclgraph_cache_tensor_constant(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x, y):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                x = x + y
                x = torch.maximum(x, torch.tensor(torch.finfo(x.dtype).min, device=x.device))
                return x

            def prompt(self, x, y):
                return self._forward(x, y)

        options = {"clone_input": False, "input_inplace_pass": True}

        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([2, 2])
        y = torch.randn([2, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(x, y)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x, y)  # cache hint
            model_match_cache(x, y)  # cache hint

    def test_aclgraph_cache_compile_with_static_kernel(self):
        class StaticKernelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

            def prompt(self, x):
                return self._forward(x)

        options = {"clone_input": False, "input_inplace_pass": True, "static_kernel_compile": True}
        
        model = StaticKernelModel()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        options = {"clone_input": False, "input_inplace_pass": True, "static_kernel_compile": False}
        model2 = StaticKernelModel()
        prompt2_cache_bin = CompiledModel.get_cache_bin(model2.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt2_cache_bin)))
        prompt2_cache_dir = os.path.abspath(os.path.dirname(prompt2_cache_bin))
        self.assertNotEqual(prompt2_cache_dir, prompt_cache_dir,
                            "Cache bin dir with different config should not be the same.")

        options = {"clone_input": False, "input_inplace_pass": True, "static_kernel_compile": True}
        model3 = StaticKernelModel()
        prompt3_cache_bin = CompiledModel.get_cache_bin(model3.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt3_cache_bin)))
        prompt3_cache_dir = os.path.abspath(os.path.dirname(prompt3_cache_bin))
        self.assertEqual(prompt3_cache_dir, prompt_cache_dir,
                            "Cache bin dir with same config and same model should be the same.")
        
    def test_aclgraph_cache_recompile_with_warning(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x, is_prompt):
                return self.cached_prompt(x, is_prompt)

            def _forward(self, x, is_prompt):
                ln1 = self.linear1(x)
                ln2 = ln1
                if is_prompt:
                    ln2 = self.linear2(x)
                return ln1 + ln2

            def prompt(self, x, is_prompt):
                return self._forward(x, is_prompt)
        options = {
            "clone_input": False,
            "input_inplace_pass": True
        }
        model = Model()
    
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([3, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x, True)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        with capture_warnings() as stdout:
            model(x, False)
        self.assertFalse(os.path.exists(prompt_cache_dir))  # recompile
        self.assertTrue(
            "UserWarning: Skip cache as" in stdout.getvalue(),
            f"Expect that UserWarning 'UserWarning: Skip cache as'"
            f"not found in logs: {stdout.getvalue()}"
        )
        self.assertTrue(
            "recompiled, set torch._logging.set_logs(recompiles=True) for details" in stdout.getvalue(),
            f"Expect that warning 'recompiled, set torch._logging.set_logs(recompiles=True) for details'"
            f"not found in logs: {stdout.getvalue()}"
        )

    @unittest.skipIf(torch.__version__ < "2.4.0", "_mark_static_inputs_attr is unsupported when torch < 2.4")
    def test_aclgraph_cache_compile_with_parameters_in_high_version_of_torch(self):
        from npugraph_ex.inference._cache_compiler import CacheBackend
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                ln1 = self.linear1(x)
                ln2 = ln1
                return ln1 + ln2

            def prompt(self, x):
                return self._forward(x)

        def check_inputs(inputs):
            has_parameter = False
            for _, input in enumerate(inputs):
                if hasattr(input, "_torchair_is_parameter"):
                    has_parameter = True
            assert has_parameter == True, f"expect cachebackend set '_torchair_is_parameter' attr to inputs, but None."

        def decorator(fw_compiler):
            def wrapper(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
                ret = fw_compiler(self, gm, example_inputs)
                check_inputs(example_inputs)
                return ret

            return wrapper

        CacheBackend.fw_compiler = decorator(CacheBackend.fw_compiler)
        options = {
            "clone_input": False,
            "input_inplace_pass": True
        }
        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))

        x = torch.randn([3, 2])

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(x)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

    def test_aclgraph_cache_compile_static_kernel_run_eagerly(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2)
                self.linear2 = torch.nn.Linear(2, 2)
                for param in self.parameters():
                    torch.nn.init.ones_(param)
                self.cached_prompt = npugraph_ex.inference.cache_compile(self.prompt, config=buildCompileConfig(options))

            def forward(self, x):
                return self.cached_prompt(x)

            def _forward(self, x):
                ln1 = self.linear1(x)
                ln2 = self.linear2(x)
                return ln1 + ln2

            def prompt(self, x):
                return self._forward(x)
        options = {
            "force_eager": True,
            "static_kernel_compile": True
        }
        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, config=buildCompileConfig(options))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        x = torch.randn([3, 2])

        self.assertFalse(os.path.exists(prompt_cache_dir))
        with capture_logger() as stdout:
            result = model(x)

        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        self.assertTrue("compile_configs[\"run_eagerly\"] = \"1\"" in stdout.getvalue())
        self.assertTrue("compile_configs[\"_aclnn_static_shape_kernel\"] = \"1\"" in stdout.getvalue())

if __name__ == '__main__':
    unittest.main()