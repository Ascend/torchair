import contextlib
import functools
import os
import tempfile
import subprocess
import sys
from typing import Dict
from dataclasses import dataclass
from pathlib import Path

import torch

from torch._inductor.codegen.common import IndentedBuffer
from inductor_npu_ext.common.debug import save_asserts
from inductor_npu_ext.common.utils import load_compiler, validate_lib, file_lock
from inductor_npu_ext import config


@dataclass
class FusedKernelSpec:
    name: str
    tiling_def: str
    host_impl: str
    device_impl: str
    cpp_wrapper: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def codegen_cpp_source(kernel_spec: FusedKernelSpec, kernel_path: str, lib_dir: str):
    wrapper = IndentedBuffer()
    wrapper.splice('''
    #include <iostream>
    #include <dlfcn.h>
    #include <cstdint>
    #include <mutex>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
    ''')

    wrapper.splice(kernel_spec.tiling_def)
    wrapper.splice("""
    const static bool debug = std::getenv("TORCH_COMPILE_DEBUG") != nullptr;
    #undef DLOG
    #define DLOG() if (debug) std::cerr << "[WRAPPER] "
    namespace {
    inline void *GetStream(void *stream) {
        return (stream == nullptr) ? c10_npu::getCurrentNPUStream().stream() : stream;
    }
    inline void *MallocWorkspace(int64_t size, void *stream) {
        return c10_npu::NPUCachingAllocator::raw_alloc_with_stream(size_t(size), stream);
    }
    inline void FreeWorkspace(void *ptr) {
        c10_npu::NPUCachingAllocator::raw_delete(ptr);
    }
    } // namespace
    """)
    wrapper.splice(kernel_spec.cpp_wrapper)

    return wrapper.getvalue()


@contextlib.contextmanager
def recover_dir():
    current_dir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(current_dir)


def never_change_dir(func):
    def wrapper(*args, **kwargs):
        with recover_dir():
            return func(*args, **kwargs)

    return wrapper


@never_change_dir
def build_ascend_lib(kernel_py):
    if config._debugging_host_only:
        with open(kernel_py, 'r', encoding='utf-8') as f:
            exec(f.read())
        return
    args = [f"{sys.executable}", kernel_py]
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to build ascend kernel, trigger bug by following command: '{' '.join(args)}'")


@never_change_dir
def _build_cpp(wrapper_cpp: str, *, compile_flags, output_file):
    args = ["g++", "-shared", "-std=c++17", "-fPIC", "-Wall", "-O2", "-o", output_file, wrapper_cpp] + compile_flags
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to build wrapper, trigger bug by following command: '{' '.join(args)}'")


def setup_asc_jit_command(spec, **kwargs):
    command_args = [f'--{k}={v}' for k, v in kwargs.items()]
    py_code = IndentedBuffer()
    py_code.splice(f"from autofuse.compile_adapter import jit_compile")
    py_code.splice(f"tiling_def = '''{spec.tiling_def}'''")
    py_code.splice(f"host_impl = '''{spec.host_impl}'''")
    py_code.splice(f"device_impl = '''{spec.device_impl}'''")
    py_code.writeline(
        f"jit_compile(tiling_def, host_impl, device_impl, {command_args})")
    return py_code.getvalue()


def is_file_content_equal(file_path: str, content: str) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content.strip() == content.strip()


def save_manual_asserts(fn, content):
    fd = os.open(fn, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, 'w+') as f:
        f.write(content)


class NpuContext:
    def __init__(self):
        self.compile_flags = []
        self.tmp_resource = None

    def __enter__(self):
        if not config._debugging_host_only:
            import torch_npu
            torch_npu_dir = os.path.dirname(torch_npu.__file__)
            ascend_dir = os.path.dirname(os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp"))
            torch_dir = os.path.dirname(torch.__file__)
            self.compile_flags = [f"-I{v}/include" for v in [ascend_dir, torch_dir, torch_npu_dir]]
            self.compile_flags.extend([f"-L{ascend_dir}/lib64", f"-lascendcl", f"-lnnopbase"])
            self.compile_flags.extend([f"-L{torch_npu_dir}/lib", f"-ltorch_npu"])
        else:
            self.tmp_resource = tempfile.TemporaryDirectory()
            self.compile_flags = [f"-I{self.tmp_resource.name}/include"]
            stub_header_dir = os.path.join(self.tmp_resource.name, "include/torch_npu/csrc/core/npu")
            os.makedirs(stub_header_dir)
            Path(os.path.join(stub_header_dir, "NPUCachingAllocator.h")).touch()
            with open(os.path.join(stub_header_dir, "NPUStream.h"), 'w') as f:
                f.write('''
                        namespace c10_npu {
                            struct getCurrentNPUStream{
                                void *stream(){ return (void*)0x123; }
                            };
                            namespace NPUCachingAllocator {
                                void *raw_alloc_with_stream(size_t size, void *stream) { return (void*)0x456; }
                                void raw_delete(void *ptr) { return; }
                            }
                        }''')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tmp_resource:
            self.tmp_resource.cleanup()


def compile_ascendc(artifacts: Dict, lib_dir: str, asserts_base: str = None, *, soc_version, force_unknow=False):
    kernel_spec = FusedKernelSpec(**artifacts)
    os.makedirs(lib_dir, exist_ok=True)

    lib_wrapper = os.path.join(lib_dir, f"wrapper.so")
    lib_kernel = os.path.join(lib_dir, f"kernel.so")

    def cache_command(cache_file, content, func, target):
        if os.path.exists(target):
            validate_lib(target)
            return
        os.remove(cache_file) if os.path.exists(cache_file) else None
        save_manual_asserts(cache_file, content)
        func(cache_file)
        validate_lib(target, change_permissions=True)

    with NpuContext() as ctx, file_lock(Path(lib_dir) / "compile.lock"):
        if os.path.exists(lib_kernel) and os.path.exists(lib_wrapper):
            validate_lib(lib_kernel)
            validate_lib(lib_wrapper)
            return

        jit_command = setup_asc_jit_command(kernel_spec, output_file=lib_kernel, graph_name=kernel_spec.name,
                                            force_unknown=force_unknow, soc_version=soc_version, config_file=None)
        save_asserts(kernel_spec.name, jit_command, 'asc_kernel.py', asserts_base)
        with load_compiler(kernel_spec.name):
            cache_command(os.path.join(lib_dir, 'asc_kernel.py'), jit_command, build_ascend_lib, lib_kernel)

        compile_comments = ' '.join(["//", "g++", "-shared", "-std=c++17", "-fPIC", "-Wall",
                                     "-O2", "-o", lib_wrapper, "inductor_wrapper.cpp"] + ctx.compile_flags + ['\n'])
        cpp_source = compile_comments + codegen_cpp_source(kernel_spec, lib_kernel, lib_dir)
        save_asserts(kernel_spec.name, cpp_source, 'inductor_wrapper.cpp', asserts_base)
        build_wrapper = functools.partial(_build_cpp, compile_flags=ctx.compile_flags, output_file=lib_wrapper)
        cache_command(os.path.join(lib_dir, 'inductor_wrapper.cpp'), cpp_source, build_wrapper, lib_wrapper)
