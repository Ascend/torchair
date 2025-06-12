import contextlib
import functools
import hashlib
import logging
import os
import re
import sys
import tempfile
import subprocess
from typing import Dict
from dataclasses import dataclass

import torch
from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.debug import save_asserts
from npu_extension_for_inductor.common.utils import load_compiler, is_kernel_need_stub


@dataclass
class FusedKernelSpec:
    name: str
    tiling_def: str
    host_impl: str
    device_impl: str
    cpp_wrapper: str


def codegen_cpp_source(kernel_spec: FusedKernelSpec, kernel_path: str):
    wrapper = IndentedBuffer()
    wrapper.splice('''
    #include <iostream>
    #include <dlfcn.h>
    #include <cstdint>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
    ''')

    wrapper.splice(kernel_spec.tiling_def)

    wrapper.writeline(f'const char *kernel_file = "{kernel_path}";')
    wrapper.splice("""
    const static bool debug = std::getenv("TORCH_COMPILE_DEBUG") != nullptr;
    #undef DLOG
    #define DLOG() if (debug) std::cerr << "[WRAPPER] "
    static void *handle = nullptr;
    static bool initialized = false;
    namespace {
    __attribute__((constructor)) void Init() {
        if (initialized) return;
        handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            std::cerr << "Failed to load " << kernel_file << ": " << dlerror() << std::endl;
            return;
        }
        DLOG() << "Kernel api lib " << kernel_file << " load succeed" << std::endl;
        initialized = true;
    }
    __attribute__((destructor)) void DeInit() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
        initialized = false;
    }
    inline void *GetFunc(const char *func_name) {
        if (handle == nullptr) {
            return nullptr;
        }
        void *func = dlsym(handle, func_name);
        if (func == nullptr) {
            std::cerr << "Failed to load api func: " << dlerror() << std::endl;
        }
        return func;
    }
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
def build_ascend_lib(jit_command):
    from types import ModuleType
    mod = ModuleType('build_kernel_mod')
    exec(compile(jit_command, '<string>', 'exec'), mod.__dict__, mod.__dict__)


@never_change_dir
def _build_cpp(source_code: str, *, compile_flags, output_file):
    with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w+', delete=True) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()
        args = ["g++", "-shared", "-std=c++17", "-fPIC", "-Wall", "-O2", "-o", output_file,
                temp_file.name] + compile_flags
        logging.debug(' '.join(args))
        subprocess.run(args, check=True)


def setup_asc_jit_command(npu_ctx: 'NpuContext', spec, **kwargs):
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
        if os.getenv("ASCIR_NOT_READY", None) != "1" or 'torch_npu' in sys.modules:
            import torch_npu
            torch_npu_dir = os.path.dirname(torch_npu.__file__)
            ascend_dir = os.path.dirname(os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp"))
            torch_dir = os.path.dirname(torch.__file__)
            self.compile_flags = [f"-I{v}/include" for v in [ascend_dir, torch_dir, torch_npu_dir]]
            self.compile_flags.extend([f"-L{ascend_dir}/lib64", f"-lascendcl", f"-lnnopbase"])
            self.compile_flags.extend([f"-L{torch_npu_dir}/lib", f"-ltorch_npu"])
        else:
            from pathlib import Path
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


def compile_ascendc(artifacts: Dict, lib_dir: str, asserts_base: str = None):
    kernel_spec = FusedKernelSpec(**artifacts)
    os.makedirs(lib_dir, exist_ok=True)

    lib_wrapper = os.path.join(lib_dir, f"wrapper.so")
    lib_kernel = os.path.join(lib_dir, f"kernel.so")

    def cache_command(cache_file, content, func, target):
        if not os.path.exists(cache_file) or not os.path.exists(target):
            save_manual_asserts(cache_file, content)
            logging.info("Compiling %s", target)
            func(content)
            return
        if is_file_content_equal(cache_file, content):
            logging.info("Cache file %s is up to date, skip recompiling %s", cache_file, target)
        else:
            logging.warning("Cache file %s for %s has been manual changed!", cache_file, target)

    with NpuContext() as ctx:
        jit_command = setup_asc_jit_command(ctx, kernel_spec, output_file=lib_kernel, graph_name=kernel_spec.name)
        save_asserts(kernel_spec.name, jit_command, 'asc_kernel.py', asserts_base)
        with load_compiler(kernel_spec.name):
            cache_command(os.path.join(lib_dir, 'asc_kernel.py'), jit_command, build_ascend_lib, lib_kernel)

        compile_comments = ' '.join(["//", "g++", "-shared", "-std=c++17", "-fPIC", "-Wall",
                                     "-O2", "-o", lib_wrapper, "inductor_wrapper.cpp"] + ctx.compile_flags + ['\n'])
        cpp_source = compile_comments + codegen_cpp_source(kernel_spec, lib_kernel)
        save_asserts(kernel_spec.name, cpp_source, 'inductor_wrapper.cpp', asserts_base)
        build_wrapper = functools.partial(_build_cpp, compile_flags=ctx.compile_flags, output_file=lib_wrapper)
        cache_command(os.path.join(lib_dir, 'inductor_wrapper.cpp'), cpp_source, build_wrapper, lib_wrapper)
