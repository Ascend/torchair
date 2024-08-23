import contextlib
import os
import re
import tempfile
from ctypes import cdll, c_size_t, c_int64, c_void_p
import subprocess
from typing import Dict
from dataclasses import dataclass

import torch
from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.debug import save_asserts


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
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    ''')

    wrapper.splice(kernel_spec.tiling_def)

    wrapper.writeline(f'const char *kernel_file = "{kernel_path}";')
    wrapper.splice("""
    const static bool debug = std::getenv("TORCH_COMPILE_DEBUG") != nullptr;
    #undef DLOG
    #define DLOG() if (debug) std::cerr
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
def build_ascend_lib(spec: FusedKernelSpec, *, output_path):
    def run_jit_command(**kwargs):
        command_args = [f'--{k}={v}' for k, v in kwargs.items()]
        py_code = IndentedBuffer()
        py_code.splice(f"from compile_adapter import jit_compile")
        py_code.splice(f"tiling_def = '''{spec.tiling_def}'''")
        py_code.splice(f"host_impl = '''{spec.host_impl}'''")
        py_code.splice(f"device_impl = '''{spec.device_impl}'''")
        py_code.writeline(
            f"jit_compile('{spec.name}', tiling_def, host_impl, device_impl, {command_args})")

        save_asserts(spec.name, py_code.getvalue(), 'asc_graph_build_kernel.py')

        from types import ModuleType
        mod = ModuleType('build_kernel_mod')
        exec(compile(py_code.getvalue(), '<string>', 'exec'), mod.__dict__, mod.__dict__)

    core_type = os.getenv("NPU_CORE_TYPE", "ascend910b1")
    lib_file = os.path.join(output_path, f"kernel_{core_type}.so")

    if os.getenv('ASCIR_NOT_READY', None) == "1":
        _build_cpp('\n'.join([spec.tiling_def, spec.host_impl, spec.device_impl]), output_file=lib_file)
        return lib_file

    run_jit_command(output_file=lib_file, soc_version=core_type)

    return lib_file


@never_change_dir
def _build_cpp(source_code: str, *, output_file):
    import torch_npu
    ascend_dir = os.path.dirname(os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp"))
    torch_dir = os.path.dirname(torch.__file__)
    torch_npu_dir = os.path.dirname(torch_npu.__file__)

    ascend_include_dir = os.path.join(ascend_dir, "include")
    torch_include_dir = os.path.join(torch_dir, "include")
    torch_npu_include_dir = os.path.join(torch_npu_dir, "include")

    extra_flags = [f"-I{v}" for v in [ascend_include_dir, torch_include_dir, torch_npu_include_dir]]
    extra_flags.extend([f"-L{ascend_dir}/lib64", f"-lascendcl", f"-lnnopbase"])
    extra_flags.extend([f"-L{torch_npu_dir}/lib", f"-ltorch_npu"])

    with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w+', delete=True) as temp_file:
        temp_file.write(source_code)
        temp_file.flush()
        args = ["g++", "-shared", "-std=c++17", "-fPIC", "-Wall", "-O2", "-o", output_file,
                temp_file.name] + extra_flags
        print(' '.join(args), flush=True)
        subprocess.run(args, check=True)


class NpuInductorKernel:
    default_stream = c_void_p(0)

    def __init__(self, wrapper_lib_path, *, name):
        self.name = name
        self.kernel = cdll.LoadLibrary(wrapper_lib_path).wrapper

    def __call__(self, *args: torch.Tensor, **sym_vals):
        result = self.kernel(*[c_void_p(t.data_ptr()) for t in args], *[c_int64(s) for s in sym_vals.values()],
                             self.default_stream)
        if result != 0:
            raise RuntimeError(f"NPU kernel {self.name} execution failed({result})")


class DummyNpuInductorKernel:
    def __init__(self, name):
        self.name = f"ACLNN_{name}"
        from npu_extension_for_inductor.common.debug import OP_SUMMARY
        self.graph_summary = OP_SUMMARY.get_graph_summary(name)
        from torch._inductor import config
        self.debug = config.trace.enabled

    def __call__(self, *args: torch.Tensor, **sym_vals):
        sym_vals = list(sym_vals.values())
        if self.graph_summary:
            args_str = [self.arg_str(arg) for arg in args]
            self.graph_summary.record_call_args(*args, sym_vals=sym_vals)
            if self.debug:
                print(f"{self.name}({','.join(args_str)}, sym_vals={sym_vals})")

    @staticmethod
    def arg_str(arg):
        if isinstance(arg, torch.Tensor):
            return f"{str(arg.dtype).split('.')[-1]}{tuple(arg.size())}"
        return str(arg)


def compile_ascendc(artifacts: Dict):
    kernel_spec = FusedKernelSpec(**artifacts)
    lib_dir = os.path.join(os.getcwd(), ".npu_kernels", kernel_spec.name)
    os.makedirs(lib_dir, exist_ok=True)

    lib_kernel = build_ascend_lib(kernel_spec, output_path=lib_dir)
    cpp_source = codegen_cpp_source(kernel_spec, lib_kernel)
    save_asserts(kernel_spec.name, cpp_source, 'inductor_wrapper.cpp')

    lib_wrapper = os.path.join(lib_dir, f"wrapper.so")
    _build_cpp(cpp_source, output_file=lib_wrapper)
    kernel = NpuInductorKernel(lib_wrapper, name=kernel_spec.name)
    return kernel
