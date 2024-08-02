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


def _get_signature(cpp_wrapper: str, kernel_name: str):
    pattern = re.compile(r'wrapper\(([^)]*)\)')
    args = [s.strip() for s in pattern.findall(cpp_wrapper)[0].split(',')][:-1]  # Remove the end stream arg
    num_addr_args = 0
    for arg in args:
        if not arg.startswith("void *"):
            break
        num_addr_args += 1
    tiling_dtype = f"{kernel_name}TilingData"
    launch_signature = ', '.join(
        ["int64_t block_dim", "void *stream"] + args[:num_addr_args] + [f"{tiling_dtype} *tiling_data"])
    tiling_signature = ', '.join(
        args[num_addr_args:] + [f"{tiling_dtype} *tiling_data", "int64_t *workspace_size", "int64_t *block_dim"])

    return tiling_signature, launch_signature


def codegen_cpp_source(kernel_spec: FusedKernelSpec, kernel_path: str):
    tiling_def = kernel_spec.tiling_def
    cpp_wrapper = kernel_spec.cpp_wrapper
    wrapper = IndentedBuffer()
    wrapper.splice('''
    #include <iostream>
    #include <dlfcn.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    ''')

    wrapper.splice(tiling_def)

    tiling_signature, launch_signature = _get_signature(cpp_wrapper, kernel_spec.name)

    wrapper.splice(f"""
    const static bool debug = std::getenv("NPU_INDUCTOR_DEBUG") != nullptr;
    #undef DLOG
    #define DLOG() if (debug) std::cerr
    static void *handle = nullptr;
    static bool initialized = false;
    typedef void (*TilingFuncType)({tiling_signature});
    typedef void (*LaunchFuncType)({launch_signature});
    static TilingFuncType tiling_fn = nullptr;
    static LaunchFuncType kernel_fn = nullptr;
    namespace {{
    __attribute__((constructor)) void Init() {{
        if (initialized) return;
        handle = dlopen("{kernel_path}", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {{
            std::cerr << "Failed to load {kernel_path}: " << dlerror() << std::endl;
            return;
        }}
        tiling_fn = reinterpret_cast<TilingFuncType>(dlsym(handle, "aclnnTiling"));
        kernel_fn = reinterpret_cast<LaunchFuncType>(dlsym(handle, "aclnnKernel"));
        if (!tiling_fn || !kernel_fn) {{
            std::cerr << "Failed to load api func: " << dlerror() << std::endl;
            return;
        }}
        DLOG() << "Kernel api lib {kernel_path} load succeed" << std::endl;
        initialized = true;
    }}
    __attribute__((destructor)) void DeInit() {{
        if (handle) {{
            dlclose(handle);
            handle = nullptr;
        }}
        initialized = false;
    }}
    }} // namespace
    """)
    wrapper.splice(cpp_wrapper)

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
