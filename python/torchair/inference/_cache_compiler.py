import ctypes
import importlib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import fcntl
import logging
import time
import types
import marshal
import os
import hashlib
from typing import List, Optional, Callable, Union, Dict, Tuple
import pickle
import shutil

import torch
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree
import torch.fx
import torch.distributed as dist

from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
from torchair.inference._gear_utils import get_dim_gears, set_dim_gears, guard_gears_shape
from torchair._utils import add_npu_patch, get_npu_default_decompositions


@dataclass
class ModelCacheMeta:
    name: str
    date: str
    version: str


@dataclass
class CompiledFX:
    signature: str
    py_code: str
    input_parameters: List[Tuple[str, bool]]


@dataclass
class ModelCacheArtifact:
    meta: ModelCacheMeta
    compiled_fn: bytes
    compiled_fx: CompiledFX


@contextmanager
def file_lock(file_descriptor, lock_type):
    fcntl.flock(file_descriptor, lock_type)
    try:
        yield
    finally:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


@contextmanager
def timer(prefix: str):
    start_time = time.time()
    yield
    logger.info("%s took %.3f [s]", prefix, time.time() - start_time)


def _compile_ge_kernel(py_code: str):
    from types import ModuleType
    ge_mod = ModuleType('ge_mod')
    exec(compile(py_code, '<string>', 'exec'), ge_mod.__dict__, ge_mod.__dict__)
    return getattr(ge_mod, 'kernel')


def _patch_user_const(code: types.CodeType):
    consts = tuple(f'<user_class>{c.__module__}|{c.__name__}' if isinstance(c, type) else c for c in code.co_consts)
    return code.replace(co_consts=consts)


def _depatch_user_const(code: types.CodeType):
    consts = []
    for c in code.co_consts:
        if isinstance(c, str) and c.startswith('<user_class>'):
            import importlib
            module, name = c[len('<user_class>'):].split('|')
            consts.append(importlib.import_module(module).__dict__[name])
        else:
            consts.append(c)
    return code.replace(co_consts=tuple(consts))


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
        self.compiled_fx: Optional[CompiledFX] = None

    def __str__(self):
        if self.compiled_fx is None:
            return f"CompiledModel({self.meta}) not compiled yet"
        return f"CompiledModel({self.meta}) with compiled fx {self.compiled_fx.signature}"

    def save(self, cache_bin: str):
        if not all([self.compiled_fn, self.compiled_fx]):
            return

        logger.info(f'Saving cache for {self.name} to {cache_bin}')
        try:
            serialized_fn = marshal.dumps(self.compiled_fn)
        except ValueError as e:
            try:
                serialized_fn = marshal.dumps(_patch_user_const(self.compiled_fn))
            except:
                logger.warning(f"Skip cache as failed to serialize compiled fn: \n{_readable_inst(self.compiled_fn)}")
                return
        artifacts = ModelCacheArtifact(meta=self.meta, compiled_fn=serialized_fn, compiled_fx=self.compiled_fx)

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

        with open(cache_bin, "rb") as f, timer(f"load cache from {cache_bin}"):
            with file_lock(f, fcntl.LOCK_SH):
                artifacts: ModelCacheArtifact = pickle.load(f)

        model = cls(artifacts.meta)
        if model.meta.version != cls.VERSION:
            raise ValueError(f"Version mismatch: {model.meta.version} != {cls.VERSION}")

        model.compiled_fn = _depatch_user_const(marshal.loads(artifacts.compiled_fn))
        model.compiled_fx = artifacts.compiled_fx
        logger.info(f"Cache {model.meta} loaded from {cache_bin}")
        return model

    @staticmethod
    def get_cache_bin(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                      cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None,
                      pp_rank: Optional[int] = None, ge_cache: bool = False, **kwargs) -> str:
        cache_dir = cache_dir or os.getenv('TORCHAIR_CACHE_HOME', os.path.join(os.getcwd(), ".torchair_cache"))
        config = config or CompilerConfig()
        if isinstance(func, types.MethodType):
            constraint = str(func.__self__) + _get_str_options(config)
            cls_name = func.__self__.__class__.__name__
        else:
            constraint = inspect.getsource(func) + _get_str_options(config)
            cls_name = func.__name__

        dist_suffixes = []
        if torch.distributed.is_available() and torch.distributed.is_initialized() and dist.get_world_size() > 1:
            global_rank = dist.get_rank() if global_rank is None else global_rank
            dist_suffixes.append(f'world{dist.get_world_size()}global_rank{global_rank}')
        if tp_rank is not None:
            dist_suffixes.append(f'tp_rank{tp_rank}')
        if pp_rank is not None:
            dist_suffixes.append(f'pp_rank{pp_rank}')
        dist_dir = '_'.join(dist_suffixes)

        trace_tag = ['dynamic'] if dynamic else ['static']
        md5 = hashlib.md5(constraint.encode()).hexdigest()

        suffixes = [cls_name] + trace_tag
        if ge_cache:
            suffixes += ['gecache']
        suffixes += [str(md5)]
        cache_bin = os.path.join(cache_dir, '_'.join(suffixes), dist_dir, func.__name__, CompiledModel.FILE)
        return os.path.abspath(cache_bin)

    def recompile(self, config: CompilerConfig):
        raise RuntimeError(f"Recompile {self} is not supported yet")

    def _get_used_params(self, model):
        log = logger if logger.isEnabledFor(logging.DEBUG) else None

        model_params = {
            **dict(model.named_parameters()),
            **dict(model.named_buffers())
        } if model else {}

        parameters = []
        for name, is_parameter in self.compiled_fx.input_parameters:
            type_str = 'parameter' if is_parameter else 'buffer'
            if log is not None:
                log.debug(f"Prefetch input {len(parameters)} from {type_str} {name}")
            if name in model_params:
                parameters.append(model_params[name])
            else:
                attr_road = name.split('.')
                if attr_road[0] != "L['self']":
                    raise ValueError(f"{type_str} {name} not supported now")
                user_buffer = eval(f"obj.{'.'.join(attr_road[1:])}", globals(), {'obj': model})
                parameters.append(user_buffer)
        return parameters

    def rebase(self, model, global_vars=None, closure=None):
        log = logger if logger.isEnabledFor(logging.DEBUG) else None
        if log is not None:
            log.debug(f"Rebasing {self.meta} onto {type(model)}")

        fn_names = [f for f in self.compiled_fn.co_names if f.startswith("__compiled_fn")]
        if len(fn_names) != 1:
            raise ValueError(f"Expected 1 compiled function, found {fn_names}")

        with timer(f"{self.name} compile ge graph"):
            ge_kernel = _compile_ge_kernel(self.compiled_fx.py_code)

        parameters = self._get_used_params(model)

        def compiled_fn(*args):
            full_args = []
            full_args.extend(parameters)
            full_args.extend(args)
            return ge_kernel(*full_args)

        global_vars = global_vars or globals()
        g = global_vars.copy()
        g.update({fn_names[0]: compiled_fn})
        for var_name in self.compiled_fn.co_names:
            if not var_name.startswith("__import_") or var_name in g:
                continue
            module_name = var_name[len("__import_"):].replace("_dot_", ".")
            logging.debug(f"Importing module {module_name} for {var_name}")
            module = importlib.import_module(module_name)
            g.update({var_name: module})
        compiled_fn = types.FunctionType(self.compiled_fn, g, closure=closure)

        if model is None:
            return compiled_fn

        def compiled_method(*args, **kwargs):
            return compiled_fn(model, *args, **kwargs)

        return compiled_method

    def readable(self, print_output=True, file: Optional[str] = None):
        readable_str = ['=' * 100, str(self.meta), str(self.compiled_fx.signature), _pretty_title('caller frame'),
                        _readable_inst(self.compiled_fn)]
        if print_output:
            print('\n'.join(readable_str))
        if file:
            abs_file = os.path.abspath(file)
            os.makedirs(os.path.dirname(abs_file), exist_ok=True)
            with open(file, 'w') as f:
                comments = '\n'.join(readable_str).split('\n')
                f.write('\n'.join([f'# {c}' for c in comments]))
                f.write('\n')
                f.write(self.compiled_fx.py_code)
                os.chmod(f.fileno(), 0o755)
        return '\n'.join(readable_str)


def _get_str_options(options: CompilerConfig, sep=","):
    g_opts, l_opts = options.as_dict()
    g_opts.update(l_opts)
    return sep.join([f"{k}={v}" for k, v in g_opts.items()])


def _pretty_title(title, length=100):
    pad = max(0, length - len(title))
    return '-' * (pad // 2) + title + '-' * (pad - pad // 2)


def _readable_inst(code):
    import dis
    import io
    output = io.StringIO()
    dis.dis(code, file=output)
    return output.getvalue()


class CacheBackend:
    def __init__(self, config: Optional[CompilerConfig], saver: 'ModelCacheSaver', *,
                 fw_compiler: Callable = None, decompositions: dict = None, extend_config: Optional[dict] = None):
        self.config = config or CompilerConfig()
        self.saver = saver
        self.extend_config = extend_config
        self.custom_decompositions = decompositions or {}
        self.input_dim_gears: Dict[int, List[int]] = dict()
        if fw_compiler is None:
            from torchair.npu_fx_compiler import get_compiler
            self.compiler = get_compiler(config)
        else:
            self.compiler = fw_compiler

    def __call__(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor], *args):
        self.saver.save_reserved_params(gm)

        for i, t in enumerate(inputs):
            dim_gears = get_dim_gears(t)
            if dim_gears is not None:
                self.input_dim_gears[i - len(inputs)] = dim_gears

        decompositions = get_npu_default_decompositions()
        decompositions.update(self.custom_decompositions)
        add_npu_patch(decompositions, self.config)
        return aot_module_simplified(gm, inputs, self.fw_compiler, self.bw_compiler,
                                     decompositions=decompositions, keep_inference_input_mutations=True)

    def fw_compiler(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        for i, dim_gears in self.input_dim_gears.items():
            set_dim_gears(example_inputs[i], dim_gears)
        guard_gears_shape(example_inputs)

        if not hasattr(self.compiler, 'codegen'):
            logger.warning(f"Skip cache as compiler {type(self.compiler)} does not support codegen")
            return self.compiler(gm, example_inputs)

        # Codegen return compiled fx directly if not support codegen
        with timer(f"{self.saver.name} codegen"):
            py_code = getattr(self.compiler, 'codegen')(gm, example_inputs, extend_config=self.extend_config)

        if not isinstance(py_code, str):
            logger.warning(f"Skip cache as compiler {type(self.compiler)} codegen return non-str {type(py_code)}")
            return py_code

        # need to create ge cache dir
        ge_cache_dir = os.path.dirname(os.path.abspath(self.saver.cache_bin))
        py_code += "\n" + f"os.makedirs('{ge_cache_dir}', exist_ok=True)"
        self.saver.save_compiled_fx(gm, example_inputs, self.config, py_code)
        with timer(f"{self.saver.name} compile ge graph"):
            return _compile_ge_kernel(py_code)

    @staticmethod
    def bw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        raise RuntimeError("Backward cache compilation is not supported yet")


class ModelCacheSaver:
    def __init__(self, func: Union[types.FunctionType, types.MethodType], cache_bin, *,
                 config: Optional[CompilerConfig] = None, dynamic: bool = True,
                 decompositions: Optional[dict] = None, ge_cache: bool = False):
        self.func = func
        self.model: Optional[torch.nn.Module] = None if isinstance(func, types.FunctionType) else func.__self__
        self.cache_bin = cache_bin
        self.cache_dir = os.path.abspath(os.path.dirname(cache_bin))
        self.compiled_model = CompiledModel(func)
        self.name = self.compiled_model.name
        extend_config = {"ge.graph_compiler_cache_dir": self.cache_dir, "ge.graph_key": "ge_cache"} if ge_cache else {}
        cache_backend = CacheBackend(config, self, decompositions=decompositions, extend_config=extend_config)
        self.compiled_func = torch.compile(func, backend=cache_backend, fullgraph=True, dynamic=dynamic)

        self._code_id = None
        self.input_parameters: List[Tuple[str, bool]] = []

        self.errors = []  # for record caching errors

    def save_reserved_params(self, gm: torch.fx.GraphModule):
        if self.model is None:
            return

        gm_params = {
            **dict(gm.named_parameters(remove_duplicate=False)),
            **dict(gm.named_buffers(remove_duplicate=False))
        }
        model_params = {
            **dict(self.model.named_parameters()),
            **dict(self.model.named_buffers())
        }
        for name, param in gm_params.items():
            found_named_param = False
            for source, model_param in model_params.items():
                if param is model_param:
                    self.input_parameters.append((source, isinstance(param, torch.nn.Parameter)))
                    found_named_param = True
                    break
            if not found_named_param:
                try:
                    source = getattr(gm, '_param_name_to_source')[name].name()
                    self.input_parameters.append((source, False))
                except AttributeError or KeyError or ValueError as e:
                    self.errors.append(f"Failed to find source for gm parameter {name} as {e}")

    def save_compiled_fx(self, fx: torch.fx.GraphModule, example_inputs: List[torch.Tensor], config: CompilerConfig,
                         py_code: str) -> object:
        placeholders = [n for n in fx.graph.nodes if n.op == 'placeholder']
        arg_signatures = []
        for i, n in enumerate(placeholders):
            if i >= len(self.input_parameters):
                arg_signatures.append(f'{i}:{n.name} UserInput({i - len(self.input_parameters)})={example_inputs[i]}')
            else:
                type_str = 'Parameter' if self.input_parameters[i][1] else 'Buffer'
                arg_signatures.append(f'{i}:{n.name} {type_str}({self.input_parameters[i][0]})={example_inputs[i]}')

        readable = [_pretty_title('graph module')]
        readable += [fx.print_readable(False)]
        readable += [_pretty_title('inputs')]
        readable += arg_signatures
        readable += [_pretty_title('compile options')]
        readable += [_get_str_options(config, sep='\n')]
        signature = '\n'.join(readable)

        logger.debug(f"Saving compiled fx {signature}")
        self.compiled_model.compiled_fx = CompiledFX(signature, py_code, self.input_parameters)

    def save_compiled_fn(self, code_id):
        if self._code_id == code_id:
            return
        if self._code_id is not None:
            logger.warning_once(
                f"Skip cache as {self.name} recompiled, set torch._logging.set_logs(recompiles=True) for details")
            self.__class__.remove_cache(self.cache_dir)
            return
        if len(self.errors) > 0:
            logger.warning(f"Skip cache {self.name} as following errors: {self.errors}")
            return
        self._code_id = code_id
        self.compiled_model.compiled_fn = ctypes.cast(code_id, ctypes.py_object).value
        self.compiled_model.save(self.cache_bin)

    def __call__(self, *args, **kwargs):
        with ModelCacheWatcher(self.func.__code__, self), torch.no_grad():
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
                 cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None,
                 pp_rank: Optional[int] = None, decompositions: Optional[dict] = None, ge_cache: bool = False):
        self.func = func
        self.config = config or CompilerConfig()
        self.dynamic = dynamic
        self.cache_dir = cache_dir
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self._compiled_model = None
        self.decompositions = decompositions
        self.ge_cache = ge_cache

    def compile(self, global_vars=None):
        cache_bin = CompiledModel.get_cache_bin(self.func, config=self.config, dynamic=self.dynamic,
                                                cache_dir=self.cache_dir, global_rank=self.global_rank,
                                                tp_rank=self.tp_rank, pp_rank=self.pp_rank, ge_cache=self.ge_cache)
        if os.path.exists(cache_bin):
            try:
                logger.info(f'Loading cache from {cache_bin}')
                compiled_model = CompiledModel.load(cache_bin)
                if compiled_model.compiled_fx is None:
                    compiled_model.recompile(self.config)
                model = self.func.__self__ if isinstance(self.func, types.MethodType) else None
                return compiled_model.rebase(model, global_vars, closure=self.func.__closure__)
            except Exception as e:
                logger.warning(f'Clear broken cache {cache_bin} as {e}')
                ModelCacheSaver.remove_cache(os.path.abspath(os.path.dirname(cache_bin)))

        logger.info(f'Compiling cache for {self.func} to {cache_bin}')
        return ModelCacheSaver(self.func, cache_bin, config=self.config, dynamic=self.dynamic,
                               decompositions=self.decompositions, ge_cache=self.ge_cache)

    def __call__(self, *args, **kwargs):
        if self._compiled_model is not None:
            return self._compiled_model(*args, **kwargs)

        self._compiled_model = self.compile(inspect.currentframe().f_back.f_globals)
        return self._compiled_model(*args, **kwargs)


def cache_compile(func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True,
                  cache_dir: Optional[str] = None, global_rank: Optional[int] = None, tp_rank: Optional[int] = None,
                  pp_rank: Optional[int] = None, custom_decompositions: Optional[dict] = None, ge_cache: bool = False,
                  **kwargs) -> Callable:
    if not isinstance(func, types.MethodType):
        raise ValueError(f"Only method can be cached now, got {func}")

    if not isinstance(func.__self__, torch.nn.Module):
        raise ValueError(f"Only torch.nn.Module method can be cached now, got {func}")

    if ge_cache and config is not None and config.experimental_config.frozen_parameter:
        raise ValueError("ge_cache and experimental_config.frozen_parameter cannot be enabled at the same time. "
                         "Please disable one of them.")

    # Lazy trigger cache load and determine the cache directory by distributed global_rank
    return LazyCompiledModel(func, config=config, dynamic=dynamic, cache_dir=cache_dir, global_rank=global_rank,
                             tp_rank=tp_rank, pp_rank=pp_rank, decompositions=custom_decompositions, ge_cache=ge_cache)


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
        with torch.no_grad():
            result = ModelCacheSaver(self.func, cache_file, config=self.config, dynamic=self.dynamic)(*args, **kwargs)
        self._reset()
        return result

    def _reset(self):
        self._called = False
        self._args = None
        self._kwargs = None
        self._ready = False


class _NoGuardCompiledFunction(_NoGuardCompiled):
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True):
        if not isinstance(func, types.FunctionType):
            raise ValueError(f"Trying cache non-function {func}")
        super().__init__(func, config=config, dynamic=dynamic)

    @classmethod
    def load(cls, cache_bin):
        cache_bin = os.path.abspath(cache_bin)
        if not os.path.exists(cache_bin):
            raise ValueError(f"Cache file {cache_bin} is not exists")
        return CompiledModel.load(cache_bin).rebase(None, global_vars=inspect.currentframe().f_back.f_globals)


class _NoGuardCompiledMethod(_NoGuardCompiled):
    def __init__(self, func, *, config: Optional[CompilerConfig] = None, dynamic: bool = True):
        if not isinstance(func, types.MethodType):
            raise ValueError(f"Trying cache non-method {func}")
        super().__init__(func, config=config, dynamic=dynamic)

    @classmethod
    def load(cls, cache_bin, *, self):
        cache_bin = os.path.abspath(cache_bin)
        if not os.path.exists(cache_bin):
            raise ValueError(f"Cache file {cache_bin} is not exists")
        return CompiledModel.load(cache_bin).rebase(self, global_vars=inspect.currentframe().f_back.f_globals)


def readable_cache(cache_bin, print_output=True, file=None):
    cache_bin = os.path.abspath(cache_bin)
    if not os.path.exists(cache_bin):
        raise ValueError(f"Cache file {cache_bin} is not exists")

    model = CompiledModel.load(cache_bin)
    return model.readable(print_output, file)
