import ctypes
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import fcntl
import logging
import types
import marshal
import os
import hashlib
from typing import List, Optional, Callable, Union, Dict
import pickle
import shutil

import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree
import torch.fx
import torch.distributed as dist

from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger


@dataclass
class ModelCacheMeta:
    name: str
    date: str
    version: str
    fx: torch.fx.GraphModule = None


@dataclass
class CompiledFX:
    options: CompilerConfig
    ge_callable: Union[types.CodeType, torch.fx.GraphModule]


@dataclass
class ModelCacheArtifact:
    meta: ModelCacheMeta
    compiled_fn: bytes
    gather_params: bytes
    compiled_fx: CompiledFX


@contextmanager
def file_lock(file_descriptor, lock_type):
    fcntl.flock(file_descriptor, lock_type)
    try:
        yield
    finally:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


class CompiledModel:
    VERSION = "1.0.0"
    FILE = "compiled_module"

    def __init__(self, meta: Union[ModelCacheMeta, types.FunctionType, types.MethodType]):
        if isinstance(meta, (types.FunctionType, types.MethodType)):
            signature = f'{meta.__qualname__}{inspect.signature(meta)}'
            meta = ModelCacheMeta(name=signature, date=str(datetime.datetime.now()), version=self.VERSION)
        self.meta = meta
        self.name = meta.name
        self.compiled_fn: Optional[types.CodeType] = None
        self.gather_params: Optional[types.CodeType] = None
        self.compiled_fx: Optional[CompiledFX] = None

    def __str__(self):
        if self.compiled_fx is None:
            return f"CompiledModel({self.meta}) not compiled yet"
        return f"CompiledModel({self.meta}) compiled with {_get_str_options(self.compiled_fx.options)}"

    def save(self, cache_bin: str):
        if not all([self.compiled_fn, self.gather_params, self.compiled_fx]):
            raise ValueError(f"Compiled model {self} is not ready to be saved")

        artifacts = ModelCacheArtifact(meta=self.meta, compiled_fn=marshal.dumps(self.compiled_fn),
                                       gather_params=marshal.dumps(self.gather_params),
                                       compiled_fx=self.compiled_fx)

        cache_bin = os.path.abspath(cache_bin)
        os.makedirs(os.path.dirname(cache_bin), exist_ok=True)
        with open(cache_bin, "wb") as f:
            with file_lock(f, fcntl.LOCK_EX):
                pickle.dump(artifacts, f)
                os.chmod(f.fileno(), 0o644)

        logger.info(f"Cache {self.meta} saved to {cache_bin}")

    @classmethod
    def load(cls, cache_bin: str):
        cache_bin = os.path.abspath(cache_bin)
        if not os.path.exists(cache_bin):
            raise ValueError(f"Cache file {cache_bin} is not exists")

        with open(cache_bin, "rb") as f:
            with file_lock(f, fcntl.LOCK_SH):
                artifacts: ModelCacheArtifact = pickle.load(f)

        model = cls(artifacts.meta)
        if model.meta.version != cls.VERSION:
            raise ValueError(f"Version mismatch: {model.meta.version} != {cls.VERSION}")

        model.compiled_fn = marshal.loads(artifacts.compiled_fn)
        model.gather_params = marshal.loads(artifacts.gather_params)
        model.compiled_fx = artifacts.compiled_fx
        logger.info(f"Cache {model.meta} loaded from {cache_bin}")
        return model

    @staticmethod
    def get_cache_bin(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                      root: Optional[str] = None, rank: Optional[int] = None, tp_rank: Optional[int] = None,
                      mp_rank: Optional[int] = None) -> str:
        root = root or os.getenv('TORCHAIR_CACHE_HOME', os.path.join(os.getcwd(), ".torchair_cache"))
        config = config or CompilerConfig()
        if isinstance(func, types.MethodType):
            constraint = str(func.__self__) + _get_str_options(config)
            cls_name = func.__self__.__class__.__name__
        else:
            constraint = inspect.getsource(func) + _get_str_options(config)
            cls_name = func.__name__

        dist_suffixes = []
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist_suffixes.append(f'rank{dist.get_rank()}' if rank is None else f'rank{rank}')
        if tp_rank is not None:
            dist_suffixes.append(f'tp_rank{tp_rank}')
        if mp_rank is not None:
            dist_suffixes.append(f'mp_rank{mp_rank}')
        dist_dir = '_'.join(dist_suffixes)

        trace_tag = ['dynamic'] if dynamic else ['static']
        md5 = hashlib.md5(constraint.encode()).hexdigest()
        cache_bin = os.path.join(root, '_'.join([cls_name] + trace_tag + [str(md5)]), dist_dir, func.__name__,
                                 CompiledModel.FILE)
        return os.path.abspath(cache_bin)

    def recompile(self, config: CompilerConfig):
        raise RuntimeError(f"Recompile {self} is not supported yet")

    def rebase(self, model, global_vars=None):
        log = logger if logger.isEnabledFor(logging.DEBUG) else None
        if log is not None:
            log.debug(f"Rebasing {self} onto {model}")

        global_vars = global_vars or globals()
        gather_params = types.FunctionType(self.gather_params, global_vars)

        fn_names = [f for f in self.compiled_fn.co_names if f.startswith("__compiled_fn")]
        if len(fn_names) != 1:
            raise ValueError(f"Expected 1 compiled function, found {fn_names}")

        parameters = gather_params(model, log)

        def compiled_fn(*args):
            full_args = []
            full_args.extend(parameters)
            full_args.extend(args)
            if log is not None:
                log.debug(f"Compiled function {self.name} called with {len(full_args)} inputs")
                for i, arg in enumerate(full_args):
                    type_str = 'param or buffer' if i < len(parameters) else 'runtime input'
                    log.debug(f"input{i}<{type_str}> = {arg}")
            result = self.compiled_fx.ge_callable(*full_args)
            if log is not None:
                log.debug(f"Compiled function {self.name} returns {result}")
            return result

        g = global_vars.copy()
        g.update({fn_names[0]: compiled_fn})
        compiled_fn = types.FunctionType(self.compiled_fn, g)

        if model is None:
            return compiled_fn

        def compiled_method(*args):
            return compiled_fn(model, *args)

        return compiled_method


def _get_str_options(options: CompilerConfig):
    g_opts, l_opts = options.as_dict()
    g_opts.update(l_opts)
    return ",".join([f"{k}={v}" for k, v in g_opts.items()])


def set_dim_gears(t: torch.Tensor, dim_gears: Dict[int, List[int]]):
    setattr(t, "dim_gears", dim_gears)


def get_dim_gears(t: torch.Tensor):
    return getattr(t, "dim_gears", None)


class CacheBackend:
    def __init__(self, config: Optional[CompilerConfig], saver: 'ModelCacheSaver', *, fw_compiler: Callable = None):
        self.config = config or CompilerConfig()
        self.saver = saver
        self.input_dim_gears: Dict[int, List[int]] = dict()
        if fw_compiler is None:
            from torchair.npu_fx_compiler import get_compiler
            self.compiler = get_compiler(config)
        else:
            self.compiler = fw_compiler

    def __call__(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor], *args):
        self.saver.save_gather_params(gm)

        for i, t in enumerate(inputs):
            dim_gears = get_dim_gears(t)
            if dim_gears is not None:
                self.input_dim_gears[i - len(inputs)] = dim_gears

        return aot_module_simplified(gm, inputs, self.fw_compiler, self.bw_compiler,
                                     keep_inference_input_mutations=True)

    def fw_compiler(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        for i, dim_gears in self.input_dim_gears.items():
            set_dim_gears(example_inputs[i], dim_gears)

        self.saver.save_compiled_fx(CompiledFX(self.config, gm))
        return self.compiler(gm, example_inputs)

    @staticmethod
    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        raise RuntimeError("Backward cache compilation is not supported yet")


class ModelCacheSaver:
    def __init__(self, func: Union[types.FunctionType, types.MethodType], cache_bin, *,
                 config: Optional[CompilerConfig] = None, dynamic: bool = True):
        self.func = func
        self.model: Optional[torch.nn.Module] = None if isinstance(func, types.FunctionType) else func.__self__
        self.cache_bin = cache_bin
        self.compiled_model = CompiledModel(func)
        self.name = self.compiled_model.name
        self.compiled_func = torch.compile(func, backend=CacheBackend(config, self), fullgraph=True, dynamic=dynamic)

        self._code_id = None

    def save_gather_params(self, gm: torch.fx.GraphModule):
        def _make_src(params):
            if params is None:
                return f'''
def gather_params(_, log):
    if log is not None:
        log.debug(f"Skip packing weights for FunctionType: {self.name}")
    return []
'''
            return f'''
def gather_params(model, log):
    named_params = {{
        **dict(model.named_parameters()),
        **dict(model.named_buffers())
    }}
    params = []
    if log is not None:
        log.debug(f"Prefetch {len(params)} param or buffers for {self.name}")
    for i, name in enumerate({repr(params)}):
        if log is not None:
            log.debug(f"Prefetch input{{i}} from model.{{name}} for {self.name}")
        params.append(named_params[name])
    return params
'''

        def _compile_gather_params(params):
            src = _make_src(params)
            functions = {}
            exec(src, globals(), functions)
            return functions['gather_params'].__code__

        if self.model is None:
            self.compiled_model.gather_params = _compile_gather_params(None)
            return

        gm_params = {
            **dict(gm.named_parameters(remove_duplicate=False)),
            **dict(gm.named_buffers(remove_duplicate=False))
        }
        flat_gm_params, _ = pytree.tree_flatten(gm_params)
        model_params = {
            **dict(self.model.named_parameters()),
            **dict(self.model.named_buffers())
        }
        ordered_param_names = []
        for param in flat_gm_params:
            for name, model_param in model_params.items():
                if param is model_param:
                    ordered_param_names.append(name)
                    break

        self.compiled_model.gather_params = _compile_gather_params(ordered_param_names)

    def save_compiled_fx(self, compiled_fx: CompiledFX):
        self.compiled_model.compiled_fx = compiled_fx

    def save_compiled_fn(self, code_id):
        if self._code_id == code_id:
            return
        if self._code_id is not None:
            logger.warning(
                f"Skipping cache as {self.name} recompiled, set torch._logging.set_logs(recompiles=True) for details")
            self.__class__.remove_cache(self.cache_bin)
            return
        self._code_id = code_id
        self.compiled_model.compiled_fn = ctypes.cast(code_id, ctypes.py_object).value
        logger.info(f'Saving cache for {self.name} to {self.cache_bin}')
        self.compiled_model.save(self.cache_bin)

    def __call__(self, *args, **kwargs):
        with ModelCacheWatcher(self.func.__code__, self), torch.inference_mode():
            return self.compiled_func(*args, **kwargs)

    @classmethod
    def remove_cache(cls, cache_bin):
        if not os.path.exists(cache_bin):
            return
        if os.path.isdir(cache_bin):
            shutil.rmtree(cache_bin)
        elif os.path.isfile(cache_bin):
            os.remove(cache_bin)
        else:
            raise RuntimeError(f'Failed remove cache {cache_bin}')


class ModelCacheWatcher:
    def __init__(self, code: types.CodeType, saver: ModelCacheSaver):
        self.origin_code = code
        self.saver = saver

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            from torch._dynamo.utils import orig_code_map
        except ImportError:
            logger.warning(f'Skip cache as can not found compiled code')
            return False

        for code_id, code in orig_code_map.values.items():
            if code == self.origin_code:
                self.saver.save_compiled_fn(code_id)
        return False


class LazyCompiledModel:
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                 root: Optional[str] = None,
                 rank: Optional[int] = None, tp_rank: Optional[int] = None,
                 mp_rank: Optional[int] = None):
        self.func = func
        self.config = config or CompilerConfig()
        self.dynamic = dynamic
        self.root = root
        self.rank = rank
        self.tp_rank = tp_rank
        self.mp_rank = mp_rank
        self._compiled_model = None

    def compile(self):
        cache_bin = CompiledModel.get_cache_bin(self.func, config=self.config, dynamic=self.dynamic, root=self.root,
                                                rank=self.rank, tp_rank=self.tp_rank, mp_rank=self.mp_rank)
        if os.path.exists(cache_bin):
            try:
                logger.info(f'Loading cache from {cache_bin}')
                compiled_model = CompiledModel.load(cache_bin)
                if compiled_model.compiled_fx is None:
                    compiled_model.recompile(self.config)
                model = self.func.__self__ if isinstance(self.func, types.MethodType) else None
                return compiled_model.rebase(model)
            except Exception as e:
                logger.warning(f'Clear broken cache {cache_bin} as {e}')
                ModelCacheSaver.remove_cache(cache_bin)

        return ModelCacheSaver(self.func, cache_bin, config=self.config, dynamic=self.dynamic)

    def __call__(self, *args, **kwargs):
        if self._compiled_model is not None:
            return self._compiled_model(*args, **kwargs)

        self._compiled_model = self.compile()
        return self._compiled_model(*args, **kwargs)


def cache_compile(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                  root: Optional[str] = None, rank: Optional[int] = None, tp_rank: Optional[int] = None,
                  mp_rank: Optional[int] = None) -> Callable:
    if not isinstance(func, types.MethodType):
        raise ValueError(f"Only method can be cached now, got {func}")

    if not isinstance(func.__self__, torch.nn.Module):
        raise ValueError(f"Only torch.nn.Module method can be cached now, got {func}")

    # Lazy trigger cache load and determine the cache directory by distributed rank
    return LazyCompiledModel(func, config=config, dynamic=dynamic, root=root, rank=rank, tp_rank=tp_rank,
                             mp_rank=mp_rank)


class _NoGuardCompiled:
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True):
        self.func = func
        self.config = config or CompilerConfig()
        self.dynamic = dynamic
        self._called = False
        self._args = None
        self._kwargs = None
        self._ready = False

    def for_inputs(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._ready = True
        return self

    def save_to(self, cache_file):
        if self._called:
            raise RuntimeError("NoGuardFunctionCacheSaver can only be called once")
        self._called = True

        if not self._ready:
            raise RuntimeError("NoGuardFunctionCacheSaver must be called with for_inputs before save_to")

        cache_file = os.path.abspath(cache_file)
        if os.path.exists(cache_file):
            raise ValueError(f"Cache file {cache_file} is already exists")

        args = self._args
        kwargs = self._kwargs
        self._args = None
        self._kwargs = None
        torch._dynamo.reset()  # reset all dynamo cache for new cache
        with torch.inference_mode():
            result = ModelCacheSaver(self.func, cache_file, config=self.config, dynamic=self.dynamic)(*args, **kwargs)
        self._reset()
        return result

    def _reset(self):
        self._called = False
        self._args = None
        self._kwargs = None
        self._ready = False


class NoGuardCompiledFunction(_NoGuardCompiled):
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True):
        if not isinstance(func, types.FunctionType):
            raise ValueError(f"Trying cache non-function {func}")
        super().__init__(func, config=config, dynamic=dynamic)

    @classmethod
    def load(cls, cache_bin):
        cache_bin = os.path.abspath(cache_bin)
        if not os.path.exists(cache_bin):
            raise ValueError(f"Cache file {cache_bin} is not exists")
        return CompiledModel.load(cache_bin).rebase(None)


class NoGuardCompiledMethod(_NoGuardCompiled):
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True):
        if not isinstance(func, types.MethodType):
            raise ValueError(f"Trying cache non-method {func}")
        super().__init__(func, config=config, dynamic=dynamic)

    @classmethod
    def load(cls, cache_bin, *, self):
        cache_bin = os.path.abspath(cache_bin)
        if not os.path.exists(cache_bin):
            raise ValueError(f"Cache file {cache_bin} is not exists")
        return CompiledModel.load(cache_bin).rebase(self)
