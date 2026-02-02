import os
import hashlib
import getpass
from typing import Dict, Optional
from ctypes import cdll, c_int64, c_void_p
from concurrent.futures import Future

import torch

from inductor_npu_ext.common import logger
from inductor_npu_ext.common.utils import validate_lib
from inductor_npu_ext import config
from torch._inductor.async_compile import AsyncCompile
from . import _compiler


class _NpuInductorKernel:
    default_stream = c_void_p(0)

    def __init__(self, wrapper, kernel=None):
        kernel = kernel if kernel is not None else wrapper.replace("wrapper.so", "kernel.so")
        self.name = self.get_kernel_name(os.path.dirname(wrapper))
        self.dl = cdll.LoadLibrary(wrapper)
        self.kernel = self.dl.wrapper
        if self.dl.init(kernel.encode('utf-8')) != 0:
            raise RuntimeError(f"NPU kernel {self.name} init failed")

    def __call__(self, *args):
        if config._sync_around_fuse_kernel:
            logger.info("Start sync previous kernel for %s", self.name)
            self.sync()
            logger.info("Succeed sync previous kernel for %s", self.name)
            logger.info("Start launch kernel %s with args %s", self.name, self.args_str(args))

        result = self.kernel(*[c_void_p(t.data_ptr()) if isinstance(t, torch.Tensor) else c_int64(t) for t in args],
                             self.default_stream)
        if result != 0:
            raise RuntimeError(f"NPU kernel {self.name} execution failed({result})")

        if config._sync_around_fuse_kernel:
            logger.info("Start sync kernel %s with args %s", self.name, self.args_str(args))
            self.sync()
            logger.info("Succeed sync kernel %s", self.name)

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
        torch.npu.synchronize()

    @staticmethod
    def get_kernel_name(path):
        normalized = os.path.normpath(path)
        folders = normalized.split(os.sep)
        return folders[-3] if len(folders) >= 3 else path


def get_lib_dir(artifacts: Dict) -> str:
    name = artifacts.get('name', 'default')
    hash_str = ''.join([v for k, v in artifacts.items() if k != 'name'])
    lib_dir = os.path.join(os.getcwd(), f".npu_kernels_{getpass.getuser()}",
                           name, hashlib.md5(hash_str.encode()).hexdigest())
    return lib_dir


class _AscendcFeature(Future):
    def __init__(self, future: Future, launcher: str):
        super().__init__()
        self.future = future
        self.launcher = launcher

    def result(self, timeout=None):
        self.future.result(timeout)
        return _NpuInductorKernel(self.launcher)


def async_compile(executor: Optional[AsyncCompile], artifacts: Dict[str, str]):
    from torch._inductor import config as inductor_config
    from inductor_npu_ext.common.debug import _get_asserts_base
    from inductor_npu_ext.common.utils import file_lock
    from pathlib import Path

    lib_dir = get_lib_dir(artifacts)
    launcher = os.path.join(lib_dir, "wrapper.so")
    kernel = os.path.join(lib_dir, "kernel.so")
    with file_lock(Path(lib_dir) / "compile.lock"):
        if os.path.exists(launcher) and os.path.exists(kernel):
            validate_lib(launcher)
            validate_lib(kernel)
            logger.debug("Cache hint for %s", launcher)
            return _NpuInductorKernel(launcher)

    asserts_base = _get_asserts_base()
    soc_version = 'cpu' if config._debugging_host_only else torch.npu.get_device_properties().name
    if inductor_config.compile_threads > 1 and executor is not None:
        logger.debug("Async compile for %s", launcher)
        future = executor.process_pool().submit(_compiler.compile_ascendc, artifacts,
                                                lib_dir, asserts_base, soc_version=soc_version)
        return _AscendcFeature(future, launcher)
    else:
        logger.debug("Sync compile for %s", launcher)
        _compiler.compile_ascendc(artifacts, lib_dir, asserts_base, soc_version=soc_version)
        return _NpuInductorKernel(launcher)
