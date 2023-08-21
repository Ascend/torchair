import sys
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if sys.platform == 'win32':
    vc_version = os.getenv('VCToolsVersion', '')
    if vc_version.startswith('14.16.'):
        CXX_FLAGS = ['/sdl']
    else:
        CXX_FLAGS = ['/sdl', '/permissive-']
else:
    CXX_FLAGS = ['-g']

USE_NINJA = os.getenv('USE_NINJA') == '1'

ext_modules = [
    CppExtension(
        'torch_npu.npu', ['torch_npu_registration/open_registration_extension.cpp'],
        extra_compile_args=CXX_FLAGS),
]
setup(
    name='torch_npu',
    packages=['torch_npu_registration/.'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)})
