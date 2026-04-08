import os
import glob
import sysconfig
from distutils.errors import CompileError
from distutils.spawn import find_executable
import torch
import torch_npu
import torch.utils.cpp_extension as cpp_extension
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
source_files = glob.glob(os.path.join(BASE_DIR, "csrc", "*.asc"), recursive=True)


def get_dependency_paths():
    python_include = sysconfig.get_config_var("INCLUDEPY")
    python_lib = sysconfig.get_config_var("LIBDIR")

    torch_include_paths = cpp_extension.include_paths()
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")

    all_include_paths = [
        *torch_include_paths,
        python_include,
        torch_npu_include,
    ]

    all_libs = [
        python_lib,
        torch_lib,
        torch_npu_lib,
    ]

    return {
        "all_includes": all_include_paths,
        "all_libs": all_libs
    }


class AscendBuildExtension(build_ext):

    def _check_bisheng_compiler(self):
        bisheng_compiler = find_executable('bisheng')
        if not bisheng_compiler:
            raise RuntimeError("bisheng command not found!")

    def build_extension(self, ext):
        self._check_bisheng_compiler()
        dep_paths = get_dependency_paths()

        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        use_cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI
        abi_value = "1" if use_cxx11_abi else "0"

        compile_cmd = [
            "bisheng",
            "-x", "asc",
            "--npu-arch=dav-2201",
            "-shared",
            "-fPIC",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi_value}",
            "-ltorch_npu", "-ltorch", "-lc10",
            *ext.sources,
            "-o", ext_fullpath,
        ]

        for include_dir in dep_paths["all_includes"]:
            compile_cmd.append(f"-I{include_dir}")

        for lib_dir in dep_paths["all_libs"]:
            compile_cmd.append(f"-L{lib_dir}")

        try:
            self.spawn(compile_cmd)
        except Exception as e:
            raise CompileError(f"{str(e)}") from e


your_ext = Extension(
    name="op_extension.custom_ops_lib",
    sources=source_files,
    language="asc",
)

setup(
    name="op_extension",
    version="0.1",
    ext_modules=[your_ext],
    packages=find_packages(),
    cmdclass={"build_ext": AscendBuildExtension},
)
