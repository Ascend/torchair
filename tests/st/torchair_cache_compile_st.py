import contextlib
import dataclasses
import functools
import logging
from typing import List
import os
import unittest

import torch
import torchair
from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig
import torchair.inference
from torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
from torchair.inference._cache_compiler import _NoGuardCompiledFunction as NoGuardCompiledFunction
from torchair.inference._cache_compiler import _NoGuardCompiledMethod as NoGuardCompiledMethod
from torchair.inference import set_dim_gears

logger.setLevel(logging.DEBUG)


class PatchAttr:
    def __init__(self, obj, attr_name, new_value):
        self.obj = obj
        self.attr_name = attr_name
        self.new_value = new_value
        self.original_value = None

    def __enter__(self):
        if hasattr(self.obj, self.attr_name):
            self.original_value = getattr(self.obj, self.attr_name)
            setattr(self.obj, self.attr_name, self.new_value)
        else:
            raise AttributeError(f"{self.obj} does not have attribute {self.attr_name}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.obj, self.attr_name, self.original_value)


def raise_exception(*args, **kwargs):
    raise Exception("Should not be called")


@contextlib.contextmanager
def forbidden_attr(obj, attr_name):
    with PatchAttr(obj, attr_name, raise_exception):
        yield


@dataclasses.dataclass
class InputMeta:
    data: torch.Tensor
    is_prompt: bool


@dataclasses.dataclass
class CustomData:
    last_hidden_state: torch.Tensor = None


class CacheCompileSt(unittest.TestCase):
    def test_cache_hint(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt)
                self.cached_decode = torchair.inference.cache_compile(self.decode)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, y)
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt)
        decode_cache_dir = CompiledModel.get_cache_bin(model.decode)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        ModelCacheSaver.remove_cache(decode_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]
        decode1 = InputMeta(torch.ones(3, 2), False), [torch.ones(3, 2)]
        decode2 = InputMeta(torch.ones(4, 2), False), [torch.ones(4, 2)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(*prompt2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled
        model(*decode2)
        self.assertTrue(os.path.exists(decode_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            model_match_cache(*prompt2)  # cache hint
            model_match_cache(*decode1)  # cache hint
            model_match_cache(*decode2)  # cache hint

    def test_cache_hint_with_kwargs(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_forward = torchair.inference.cache_compile(self.raw_forward)

            def forward(self, x: torch.Tensor):
                return self.cached_forward(x=x)

            def raw_forward(self, x: torch.Tensor):
                return self.linear(x)

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model.raw_forward)
        ModelCacheSaver.remove_cache(cache_dir)

        prompt = torch.ones(3, 2)

        self.assertFalse(os.path.exists(cache_dir))
        model(prompt)
        self.assertTrue(os.path.exists(cache_dir))  # cache compiled
        model(prompt)
        self.assertTrue(os.path.exists(cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(prompt)  # cache hint
            model_match_cache(prompt)  # cache hint

    def test_cache_hint_with_explicit_kwargs(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_forward = torchair.inference.cache_compile(self.raw_forward)

            def forward(self, *, x: torch.Tensor):
                return self.cached_forward(x=x)

            def raw_forward(self, *, x: torch.Tensor):
                return self.linear(x)

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model.raw_forward)
        ModelCacheSaver.remove_cache(cache_dir)

        prompt = torch.ones(3, 2)

        self.assertFalse(os.path.exists(cache_dir))
        model(x=prompt)
        self.assertTrue(os.path.exists(cache_dir))  # cache compiled
        model(x=prompt)
        self.assertTrue(os.path.exists(cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x=prompt)  # cache hint
            model_match_cache(x=prompt)  # cache hint

    def test_cache_hint_for_complex_io_process(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt)
                self.cached_decode = torchair.inference.cache_compile(self.decode)

            def forward(self, x: InputMeta, y: List[torch.Tensor], z, s1, s2):
                if x.is_prompt:
                    return self.cached_prompt(x, y, z, s1, s2)
                return self.cached_decode(x, y, z, s1, s2)

            def _forward(self, x, y, z, s1, s2):
                mm1 = self.linear1(x.data) + self.linear2(y[0])
                sum1 = z + mm1.sum()
                ones1 = torch.ones([s1, s2]).view(-1)
                add1 = sum1 + ones1 + s2
                return (add1, add1.shape[0], 2 * s1, y[0].view(2, -1))

            def prompt(self, x, y, z, s1, s2):
                return self._forward(x, y, z, s1, s2)

            def decode(self, x, y, z, s1, s2):
                return self._forward(x, y, z, s1, s2)

        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt)
        decode_cache_dir = CompiledModel.get_cache_bin(model.decode)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        ModelCacheSaver.remove_cache(decode_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)], torch.randn([6, 2])[:, 0], 2, 3
        decode1 = InputMeta(torch.ones(3, 2), False), [torch.ones(3, 2)], torch.randn([6, 2])[:, 0], 2, 3

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))
        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            model_match_cache(*decode1)  # cache hint

    def test_forbidden_backward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt)
                self.cached_decode = torchair.inference.cache_compile(self.decode)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, y)
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        loss = model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        self.assertRaises(Exception, loss.backward)

    def test_skip_cache_as_recompile(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, dynamic=False)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, y)
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear(x.data) + self.linear(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, dynamic=False)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(*prompt2)
        self.assertFalse(os.path.exists(prompt_cache_dir))  # recompile and clear cache

    def test_no_guard_method(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

            def forward(self, x, y):
                return self.linear(x.data) + self.linear(y[0])

        model = Model()

        cache1 = 'model_prompt_3_2'
        cache2 = 'model_prompt_4_2'
        ModelCacheSaver.remove_cache(cache1)
        ModelCacheSaver.remove_cache(cache2)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(4, 2), False), [torch.ones(4, 2)]

        NoGuardCompiledMethod(model.forward, dynamic=False).for_inputs(*prompt1).save_to(cache1)
        NoGuardCompiledMethod(model.forward, dynamic=False).for_inputs(*prompt2).save_to(cache2)

        self.assertTrue(os.path.exists(cache1))
        self.assertTrue(os.path.exists(cache2))

        torchair.inference.readable_cache(cache1)

        readable_file = 'cache2_readable.py'
        ModelCacheSaver.remove_cache(readable_file)
        torchair.inference.readable_cache(cache2, print_output=False, file=readable_file)
        self.assertTrue(os.path.exists(readable_file))

        NoGuardCompiledMethod.load(cache1, self=model)(*prompt1)  # assert not raise
        NoGuardCompiledMethod.load(cache2, self=model)(*prompt2)  # assert not raise

    def test_no_guard_function(self):
        def func(x, y):
            return torch.add(x, y)

        prompt1 = [torch.ones(3, 2), torch.ones(3, 2)]
        prompt2 = [torch.ones(4, 2), torch.ones(4, 2)]

        cache1 = 'func_prompt_3_2'
        cache2 = 'func_prompt_4_2'
        ModelCacheSaver.remove_cache(cache1)
        ModelCacheSaver.remove_cache(cache2)

        NoGuardCompiledFunction(func, dynamic=False).for_inputs(*prompt1).save_to(cache1)
        NoGuardCompiledFunction(func, dynamic=False).for_inputs(*prompt2).save_to(cache2)

        self.assertTrue(os.path.exists(cache1))
        self.assertTrue(os.path.exists(cache2))

        NoGuardCompiledFunction.load(cache1)(*prompt1)  # assert not raise
        NoGuardCompiledFunction.load(cache2)(*prompt2)  # assert not raise

    def test_cache_with_explicit_module(self):
        torchair.foo_tensor = torch.ones(3, 2)

        def func(x):
            import torchair
            return torch.add(x, torchair.foo_tensor)

        prompt = [torch.ones(3, 2)]

        cache = 'func_with_explicit_module'
        ModelCacheSaver.remove_cache(cache)

        NoGuardCompiledFunction(func, dynamic=False).for_inputs(*prompt).save_to(cache)

        self.assertTrue(os.path.exists(cache))

        online_only_keys = [k for k in globals().keys() if k.startswith('__import_')]
        for k in online_only_keys:
            globals().pop(k)
        NoGuardCompiledFunction.load(cache)(*prompt)  # assert not raise

    def test_use_outer_globals(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.compiled = torchair.inference.cache_compile(self._forward)

            def forward(self, x):
                return self.compiled(x)

            def _forward(self, x):
                x = torch.abs(x)
                return CustomData(
                    last_hidden_state=x
                )

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model._forward)
        ModelCacheSaver.remove_cache(cache_dir)

        x = torch.ones(1, 1, 2)

        self.assertFalse(os.path.exists(cache_dir))
        model(x)
        self.assertTrue(os.path.exists(cache_dir))  # cache compiled

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x)  # cache hint

    def test_huggingface_dataclass(self):
        try:
            import transformers.file_utils
        except:
            print("Skip test_huggingface_dataclass as transformers is not installed")
            return

        def f(x):
            from transformers.modeling_outputs import BaseModelOutputWithPast
            x = torch.add(x, x)
            return BaseModelOutputWithPast(x)

        cache_file = 'test_huggingface_dataclass'
        ModelCacheSaver.remove_cache(cache_file)
        NoGuardCompiledFunction(f).for_inputs(torch.ones(2)).save_to(cache_file)
        self.assertTrue(os.path.exists(cache_file))
        NoGuardCompiledFunction.load(cache_file)(torch.ones(2))

    def test_func_use_closure(self):
        y = torch.ones(2)
        z = torch.ones(2)

        def closure_func(x):
            return torch.add(x, z)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.compiled = torchair.inference.cache_compile(self._forward)

            def forward(self, x):
                return self.compiled(x)

            def _forward(self, x):
                return torch.add(x, closure_func(y))

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model._forward)
        ModelCacheSaver.remove_cache(cache_dir)

        x = torch.ones(2)

        self.assertFalse(os.path.exists(cache_dir))
        model(x)
        self.assertTrue(os.path.exists(cache_dir))  # cache compiled

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(x)  # cache hint

    def test_cache_hint_for_anonymous_buffer(self):
        class AnonymousModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buffer = torch.ones(2, 2)

            def forward(self, x):
                return torch.add(x, self.buffer)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anonymous = AnonymousModule()
                self.anonymous_buffer = torch.ones(2, 2)
                self.cached_forward = torchair.inference.cache_compile(self._forward)

            def forward(self, x):
                return self.cached_forward(x)

            def _forward(self, x):
                return self.anonymous(torch.add(x, self.anonymous_buffer))

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model._forward)
        ModelCacheSaver.remove_cache(cache_dir)

        prompt1 = torch.ones(2, 2),
        prompt2 = torch.ones(2, 2),
        model(*prompt1)
        model(*prompt2)

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            model_match_cache(*prompt2)  # cache hint

    def test_cache_hint_for_anonymous_buffer_with_comma(self):
        class AnonymousModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buffer = torch.ones(2, 2)

            def forward(self, x):
                return torch.add(x, self.buffer)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [AnonymousModule()]  # buffer name will be self.layers[0].buffer
                self.cached_forward = torchair.inference.cache_compile(self.raw_forward)

            def forward(self, x):
                return self.cached_forward(x)

            def raw_forward(self, x):
                return self.layers[0](torch.add(x, x))

        model = Model()

        cache_dir = CompiledModel.get_cache_bin(model.raw_forward)
        ModelCacheSaver.remove_cache(cache_dir)

        prompt1 = torch.ones(2, 2),
        prompt2 = torch.ones(2, 2),
        model(*prompt1)
        model(*prompt2)

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            model_match_cache(*prompt2)  # cache hint

    def test_decomp(self):
        from torch.library import Library
        npu_define_lib = Library("test", "DEF")
        op_name = npu_define_lib.define("add(Tensor input) -> Tensor")

        def add_cpu(t):
            return t

        def add_meta(t):
            return t.new_empty(t.size())

        npu_define_lib.impl(op_name, add_cpu, 'CPU')
        npu_define_lib.impl(op_name, add_meta, 'Meta')

        from torch._decomp import get_decompositions, register_decomposition

        @register_decomposition(torch.ops.test.add.default)
        def test_add_decomp(self):
            return self * 3

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = torchair.inference.cache_compile(self.inner_forward,
                                                               custom_decompositions=get_decompositions(
                                                                   [torch.ops.test.add.default]))

            def inner_forward(self, tensor):
                return torch.ops.test.add(tensor)

            def forward(self, tensor):
                return self.cached(tensor)

        decom_model = Model()
        t = torch.ones(1)
        cache_dir = CompiledModel.get_cache_bin(decom_model.inner_forward)
        ModelCacheSaver.remove_cache(cache_dir)
        decom_model(t)
        self.assertTrue(os.path.exists(cache_dir))
        decom_model(t)  # cache hint

    def test_cache_hint_gears(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt)
                self.cached_decode = torchair.inference.cache_compile(self.decode)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, y)
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        model = Model()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt)
        decode_cache_dir = CompiledModel.get_cache_bin(model.decode)
        ModelCacheSaver.remove_cache(prompt_cache_dir)
        ModelCacheSaver.remove_cache(decode_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        set_dim_gears(prompt1[0].data, {0: [2, 3]})
        set_dim_gears(prompt1[1][0], {0: [2, 3]})
        prompt2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]
        decode1 = InputMeta(torch.ones(3, 2), False), [torch.ones(3, 2)]
        set_dim_gears(decode1[0].data, {0: [3, 4]})
        set_dim_gears(decode1[1][0], {0: [3, 4]})
        decode2 = InputMeta(torch.ones(4, 2), False), [torch.ones(4, 2)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        model(*prompt2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled
        model(*decode2)
        self.assertTrue(os.path.exists(decode_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            model_match_cache(*prompt2)  # cache hint
            model_match_cache(*decode1)  # cache hint
            model_match_cache(*decode2)  # cache hint


    def test_ge_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, ge_cache=True)
                self.cached_decode = torchair.inference.cache_compile(self.decode, ge_cache=True)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, y)
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        model = Model()

        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt, ge_cache=True)
        decode_cache_bin = CompiledModel.get_cache_bin(model.decode, ge_cache=True)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(decode_cache_bin)))

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]
        decode1 = InputMeta(torch.ones(3, 2), False), [torch.ones(3, 2)]
        decode2 = InputMeta(torch.ones(4, 2), False), [torch.ones(4, 2)]

        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))

        decode_cache_dir = os.path.abspath(os.path.dirname(decode_cache_bin))

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        prompt2_res = model(*prompt2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled
        decode2_res = model(*decode2)
        self.assertTrue(os.path.exists(decode_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            prompt2_cache_res = model_match_cache(*prompt2)  # cache hint
            model_match_cache(*decode1)  # cache hint
            decode2_cache_res = model_match_cache(*decode2)  # cache hint
        self.assertTrue(prompt2_res.equal(prompt2_cache_res))
        self.assertTrue(decode2_res.equal(decode2_cache_res))

    def test_empty_tensor_option(self):

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached_forward = torchair.inference.cache_compile(self.raw_forward, dynamic=True)

            def forward(self, x, y):
                return self.cached_forward(x, y)

            def raw_forward(self, x, y):
                return torch.add(x, y)

        model = Model()
        cache_dir = CompiledModel.get_cache_bin(model.raw_forward)
        ModelCacheSaver.remove_cache(cache_dir)
        x1 = torch.zeros(0, 10, 10, dtype=torch.float32)
        y1 = torch.zeros(0, 10, 10, dtype=torch.float32)
        model(x1, y1)
        file_path = "test_empty_tensor_option.py"
        torchair.inference.readable_cache(cache_dir, file=file_path)
        self.assertTrue(os.path.exists(file_path), f"File does not exist: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        target = 'local_compile_options["ge.exec.allTensorNotEmpty"] = "'
        start_idx = content.find(target)
        self.assertNotEqual(start_idx, -1, f"Target string not found: {target}")
        start_idx += len(target)
        end_idx = content.find('"', start_idx)
        self.assertNotEqual(end_idx, -1, f"Missing quotes")
        value = content[start_idx:end_idx]
        self.assertEqual(value, "0", f"Value should be '0', but got {value}")

    def test_codegen_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = torchair.inference.cache_compile(self._forward)

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)

        model = Model()
        cache_dir = CompiledModel.get_cache_bin(model._forward)
        ModelCacheSaver.remove_cache(cache_dir)
        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]
        model(t1, t2, t3, s1, s2)

        code = '''
_is_first_run = True
def kernel(*args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    s0 = arg0_1
    s1 = arg2_1
    s3 = arg5_1
    s4 = arg6_1
    s5 = arg7_1
    ge_inputs = list(args)
    del ge_inputs[7]
    del ge_inputs[6]
    del ge_inputs[2]
    del ge_inputs[0]
    ge_inputs[1] = args[3].clone()
    ge_inputs[3] = torch.from_numpy(numpy.array([args[5]]))
    ge_inputs.insert(4, torch.from_numpy(numpy.array([args[6], args[7], ])))

    global _is_first_run
    if _is_first_run:
        _is_first_run = False
        assert_size_stride(args[1], (1, s0), (s0, 1))
        assert_size_stride(args[3], (s1, 1), (s4 + s5, 1))
        assert_size_stride(args[4], (s4 + s5, s1), (s1, 1))
        _update_constplaceholder_attr_from_inputs(ge_graph, args)
        _update_internal_format_from_inputs(ge_graph, ge_inputs)
        ge_graph.load(local_compile_options, create_pg=False)
        ge_graph.compile()

    ge_outputs = ge_graph.run(ge_inputs)
    fx_outputs = [None] * 4
    fx_outputs[0] = ge_outputs[0]
    fx_outputs[1] = ge_outputs[1]
    fx_outputs[2] = torch.as_strided(args[4], [s4, s1], [s1, 1], 0)
    fx_outputs[3] = torch.as_strided(args[4], [s5, s1], [s1, 1], s1*s4)

    del ge_outputs
    return tuple(fx_outputs)
'''
        compile_model = CompiledModel.load(cache_dir)
        print(compile_model.compiled_fx.py_code)
        self.assertTrue(code in compile_model.compiled_fx.py_code)

    def test_codegen_static(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cached = torchair.inference.cache_compile(self._forward, dynamic=False)

            def forward(self, t1, t2, t3, s1, s2):
                return self.cached(t1, t2, t3, s1, s2)

            def _forward(self, t1, t2, t3, s1, s2):
                return t1 + s1, t2 + 1, torch.split(t3, s2)

        model = Model()
        cache_dir = CompiledModel.get_cache_bin(model._forward, dynamic=False)
        ModelCacheSaver.remove_cache(cache_dir)
        t1 = torch.zeros(1, 10)
        t2 = torch.zeros(2, 5)[:, 0:1]
        t3 = torch.zeros(5, 2)
        s1 = 5
        s2 = [2, 3]
        model(t1, t2, t3, s1, s2)

        code = '''
_is_first_run = True
def kernel(*args):
    ge_inputs = list(args)
    ge_inputs[1] = args[1].clone()

    global _is_first_run
    if _is_first_run:
        _is_first_run = False
        assert_size_stride(args[0], (1, 10), (10, 1))
        assert_size_stride(args[1], (2, 1), (5, 1))
        assert_size_stride(args[2], (5, 2), (2, 1))
        _update_constplaceholder_attr_from_inputs(ge_graph, args)
        _update_internal_format_from_inputs(ge_graph, ge_inputs)
        ge_graph.load(local_compile_options, create_pg=False)
        ge_graph.compile()

    ge_outputs = ge_graph.run(ge_inputs)
    fx_outputs = [None] * 4
    fx_outputs[0] = ge_outputs[0]
    fx_outputs[1] = ge_outputs[1]
    fx_outputs[2] = torch.as_strided(args[2], [2, 2], [2, 1], 0)
    fx_outputs[3] = torch.as_strided(args[2], [3, 2], [2, 1], 4)

    del ge_outputs
    return tuple(fx_outputs)
'''
        compile_model = CompiledModel.load(cache_dir)
        print(compile_model.compiled_fx.py_code)
        self.assertTrue(code in compile_model.compiled_fx.py_code)

    def test_backend_params(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                config = CompilerConfig()
                config.export.experimental.enable_lite_export = True
                config.debug.data_dump._path = 'test'
                config.dump_config.dump_layer = "Add"
                npu_backend = torchair.get_npu_backend(compiler_config=config)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, config=config,
                                                                      backend=npu_backend)
                self.cached_decode = torchair.inference.cache_compile(self.decode, config=config,
                                                                      backend=npu_backend)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

            def _forward(self, x, kv):
                return self.linear2(x.data) + self.linear2(kv[0])

        model = Model()
        prompt_cache_bin = CompiledModel.get_cache_bin(model.prompt)
        decode_cache_bin = CompiledModel.get_cache_bin(model.decode)
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(prompt_cache_bin)))
        ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(decode_cache_bin)))
        prompt_cache_dir = os.path.abspath(os.path.dirname(prompt_cache_bin))
        decode_cache_dir = os.path.abspath(os.path.dirname(decode_cache_bin))

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]
        decode1 = InputMeta(torch.ones(3, 2), False), [torch.ones(3, 2)]
        decode2 = InputMeta(torch.ones(4, 2), False), [torch.ones(4, 2)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled
        prompt2_res = model(*prompt2)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # not recompile

        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled
        decode2_res = model(*decode2)
        self.assertTrue(os.path.exists(decode_cache_dir))  # not recompile

        model_match_cache = Model()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            model_match_cache(*prompt1)  # cache hint
            prompt2_cache_res = model_match_cache(*prompt2)  # cache hint
            model_match_cache(*decode1)  # cache hint
            decode2_cache_res = model_match_cache(*decode2)  # cache hint
        self.assertTrue(prompt2_res.equal(prompt2_cache_res))
        self.assertTrue(decode2_res.equal(decode2_cache_res))


    def test_backend_params_with_exception(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                config = CompilerConfig()
                config.export.experimental.enable_lite_export = True
                config1 = CompilerConfig()
                config1.export.experimental.enable_lite_export = False
                config1.debug.data_dump._path = 'test'
                config1.dump_config.dump_layer = "Add"
                backend = torchair.get_npu_backend(compiler_config=config)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, backend=backend)
                self.cached_decode = torchair.inference.cache_compile(self.decode, config=config1,
                                                                      backend=backend)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

            def _forward(self, x, kv):
                return self.linear2(x.data) + self.linear2(kv[0])

        x = InputMeta(data=torch.randn(2, 2), is_prompt=True)
        kv = [torch.randn(2, 2)]
        with self.assertRaises(ValueError) as cm:
            model = Model()
        exception = cm.exception
        self.assertEqual(str(exception),
                         "config in current backend is different from the config during cache generation.")


    def test_cache_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt, dynamic=False)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)


        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt, dynamic=False)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        prompt2 = InputMeta(torch.ones(12, 12), True), [torch.ones(12, 12)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(*prompt2)  # cache hint
            exception = cm.exception
            self.assertEqual(str(exception), "expected size 12==3, stride 12==2 at dim=0")


    def test_cache_dynamic_assert_size_stride(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_prompt = torchair.inference.cache_compile(self.prompt)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                return self.cached_prompt(x, y)

            def _forward(self, x, y):
                return self.linear2(x.data) + self.linear2(y[0])

            def prompt(self, x, y):
                return self._forward(x, y)


        model = CacheModel()

        prompt_cache_dir = CompiledModel.get_cache_bin(model.prompt)
        ModelCacheSaver.remove_cache(prompt_cache_dir)

        prompt1 = InputMeta(torch.ones(12, 2), True), [torch.ones(12, 2)]
        prompt2 = InputMeta(torch.ones(12, 12), True), [torch.ones(12, 12)]

        self.assertFalse(os.path.exists(prompt_cache_dir))
        model(*prompt1)
        self.assertTrue(os.path.exists(prompt_cache_dir))  # cache compiled

        model_match_cache = CacheModel()
        with forbidden_attr(ModelCacheSaver, '__call__'):
            with self.assertRaises(AssertionError) as cm:
                model_match_cache(*prompt2)  # cache hint
            exception = cm.exception
            self.assertEqual(str(exception), "expected size 12==12, stride 12==2 at dim=0")


    def test_cache_assert_size_stride_remove_cache(self):
        class CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)

                self.cached_decode = torchair.inference.cache_compile(self.decode, dynamic=False)

            def forward(self, x: InputMeta, y: List[torch.Tensor]):
                return self.cached_decode(x, y)

            def _forward(self, x, y):
                return self.linear(x.data) + self.linear(y[0])

            def decode(self, x, y):
                return self._forward(x, y)


        model = CacheModel()

        decode_cache_dir = CompiledModel.get_cache_bin(model.decode, dynamic=False)
        ModelCacheSaver.remove_cache(decode_cache_dir)

        decode1 = InputMeta(torch.ones(3, 2), True), [torch.ones(3, 2)]
        decode2 = InputMeta(torch.ones(2, 2), True), [torch.ones(2, 2)]

        self.assertFalse(os.path.exists(decode_cache_dir))
        model(*decode1)
        self.assertTrue(os.path.exists(decode_cache_dir))  # cache compiled
        model(*decode2)
        self.assertFalse(os.path.exists(decode_cache_dir))  # recompile

    def test_rng_check(self):
        class Model(torch.nn.Module):
            def __init__(self, dynamic):
                super().__init__()
                self.cached = torchair.inference.cache_compile(self._forward, dynamic=dynamic)

            def forward(self, x):
                return self.cached(x)

            def _forward(self, x):
                y = torch.randn(x.shape)
                return x + y

        x = torch.zeros(2, 2)

        for dynamic in [True, False]:
            model = Model(dynamic)
            cache_dir = CompiledModel.get_cache_bin(model._forward, dynamic=dynamic)
            ModelCacheSaver.remove_cache(cache_dir)
            with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed) as cm:
                model(x)
            self.assertTrue("Cache compile dose not support operator that depend on RNG, input index: 1." in str(
                cm.exception.inner_exception))


if __name__ == '__main__':
    unittest.main()

