import os
import sys
import hashlib
import logging
from typing import Dict, Callable, Optional
from ctypes import cdll, c_size_t, c_int64, c_void_p
from concurrent.futures import Future
import torch
from torch._inductor.codecache import AsyncCompile
from . import _aclnn_compiler


class _NpuInductorKernel:
    default_stream = c_void_p(0)

    def __init__(self, wrapper_lib_path):
        self.name = self.get_kernel_name(wrapper_lib_path)
        self.kernel = cdll.LoadLibrary(wrapper_lib_path).wrapper
        self.debug = os.getenv("NPU_INDUCTOR_DEBUG_SINGLE_KERNEL", None) == '1'

    def __call__(self, *args):
        if self.debug:
            logging.info("Start sync previous kernel for %s", self.name)
            self.sync()
            logging.info("Succeed sync previous kernel for %s", self.name)
            logging.info("Start launch kernel %s with args %s", self.name, self.args_str(args))

        result = self.kernel(*[c_void_p(t.data_ptr()) if isinstance(t, torch.Tensor) else c_int64(t) for t in args],
                             self.default_stream)
        if result != 0:
            raise RuntimeError(f"NPU kernel {self.name} execution failed({result})")

        if self.debug:
            logging.info("Start sync kernel %s", self.name)
            self.sync()
            logging.info("Succeed sync kernel %s", self.name)

    @staticmethod
    def args_str(args):
        def _tensor_str(t):
            if isinstance(t, torch.Tensor):
                return (f'Tensor(dtype={t.dtype}, '
                        f'shape={tuple(t.size())}, '
                        f'stride={t.stride()}, '
                        f'offset={t.storage_offset()}, '
                        f'data={hex(t.data_ptr())}, '
                        f'device={t.device})')
            return str(t)
        return ', '.join([_tensor_str(arg) for arg in args])

    @staticmethod
    def sync():
        if os.getenv("ASCIR_NOT_READY", None) != '1' or 'torch_npu' in sys.modules:
            torch.npu.synchronize()

    @staticmethod
    def get_kernel_name(path):
        normalized = os.path.normpath(path)
        folders = normalized.split(os.sep)
        return folders[-3] if len(folders) >= 3 else path


class _NpuInductorPgo:
    default_stream = c_void_p(0)

    def __init__(self, pgo_lib_path):
        self.name = self.get_kernel_name(pgo_lib_path)
        self.kernel = cdll.LoadLibrary(pgo_lib_path).pgo

    def __call__(self, *args):
        result = self.kernel(*[c_void_p(t.data_ptr()) if isinstance(t, torch.Tensor) else c_int64(t) for t in args],
                             self.default_stream)
        if result != 0:
            raise RuntimeError(f"NPU pgo kernel {self.name} execution failed({result})")

    @staticmethod
    def get_kernel_name(path):
        normalized = os.path.normpath(path)
        folders = normalized.split(os.sep)
        return folders[-3] if len(folders) >= 3 else path


def get_lib_dir(artifacts: Dict) -> str:
    name = artifacts.get('name', 'default')
    hash_str = ''.join([v for k, v in artifacts.items() if k != 'name' and k != 'pgo'])
    lib_dir = os.path.join(os.getcwd(), ".npu_kernels", name, hashlib.md5(hash_str.encode()).hexdigest())
    return lib_dir


def _get_wrapper_lib(artifacts: Dict) -> str:
    lib_dir = get_lib_dir(artifacts)
    return os.path.join(lib_dir, "wrapper.so")


def _get_pgo_lib(artifacts: Dict) -> str:
    lib_dir = get_lib_dir(artifacts)
    return os.path.join(lib_dir, "pgo.so")


class _AscendcFeature(Future):
    def __init__(self, future: Future, launcher: str):
        super().__init__()
        self.future = future
        self.launcher = launcher

    def result(self, timeout=None):
        self.future.result(timeout)
        return _NpuInductorKernel(self.launcher)


def async_compile(executor: Optional[AsyncCompile], artifacts: Dict[str, str]):
    from torch._inductor import config
    from npu_extension_for_inductor.common.debug import _get_asserts_base
    launcher = _get_wrapper_lib(artifacts)
    asserts_base = _get_asserts_base()
    if config.compile_threads > 1 and executor is not None:
        logging.info("Async compile for %s", launcher)
        future = executor.process_pool().submit(_aclnn_compiler.compile_ascendc,
                                                artifacts, os.path.dirname(launcher), asserts_base)
        return _AscendcFeature(future, launcher)
    else:
        logging.info("Sync compile for %s", launcher)
        _aclnn_compiler.compile_ascendc(artifacts, os.path.dirname(launcher), asserts_base)
        return _NpuInductorKernel(launcher)


def async_compile_pgo(executor: Optional[AsyncCompile], artifacts: Dict[str, str]):
    from torch._inductor import config
    from npu_extension_for_inductor.common.debug import _get_asserts_base
    launcher = _get_pgo_lib(artifacts)
    asserts_base = _get_asserts_base()

    logging.info("PGO compile for %s", launcher)
    _aclnn_compiler.compile_ascendc(artifacts, os.path.dirname(launcher), asserts_base, True, True)
    return _NpuInductorPgo(launcher)
