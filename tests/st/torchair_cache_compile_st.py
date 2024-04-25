import contextlib
import dataclasses
from typing import List
import os
import unittest

import torch
import torchair
import torchair.inference
from torchair.inference.cache_compiler import CompiledModel, ModelCacheSaver
from torchair.inference.cache_compiler import NoGuardCompiledFunction, NoGuardCompiledMethod


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


if __name__ == '__main__':
    unittest.main()
