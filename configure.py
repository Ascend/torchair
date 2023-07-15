"""Basic configurations for building torchair"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys

_COMPAT_TORCH_VERSION = "2."
_PYTHON_BIN_PATH_ENV = "TARGET_PYTHON_PATH"
_ASCEND_SDK_ENV = "ASCEND_SDK_PATH"
_NO_ASCEND_SDK = "NO_ASCEND_SDK"


def run_command(cmd):
    """Execute command"""
    output = subprocess.check_output(cmd)
    return output.decode('UTF-8').strip()


def get_input(question):
    """Get response from user keyboard input"""
    try:
        try:
            answer = raw_input(question)
        except NameError:
            answer = input(question)
    except EOFError:
        answer = ''
    return answer


def real_config_path(file):
    """Get complete file path"""
    return os.path.join("tools", file)


def setup_python(env_path):
    """Get python install path."""
    default_python_bin_path = sys.executable
    ask_python_bin_path = ('Please specify the location of python with valid '
                           'torch 2.x site-packages installed. [Default '
                           'is %s]\n(You can make this quiet by set env '
                           '[TARGET_PYTHON_PATH]): ') % default_python_bin_path
    custom_python_bin_path = env_path
    while True:
        if not custom_python_bin_path:
            python_bin_path = get_input(ask_python_bin_path)
        else:
            python_bin_path = custom_python_bin_path
            custom_python_bin_path = None
        if not python_bin_path:
            python_bin_path = default_python_bin_path
        # Check if the path is valid
        if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
            pass
        elif not os.path.exists(python_bin_path):
            print('Invalid python path: %s cannot be found.' % python_bin_path)
            continue
        else:
            print('%s is not executable.  Is it the python binary?' %
                  python_bin_path)
            continue

        try:
            compile_args = run_command([
                python_bin_path, '-c',
                '''
import os
import distutils.sysconfig
import torch
print('|'.join([
    torch.__version__,
    os.path.dirname(torch.__file__),
    f"-D_GLIBCXX_USE_CXX11_ABI={'1' if torch.compiled_with_cxx11_abi() else '0'}",
    distutils.sysconfig.get_python_inc()
]))
'''
            ]).split("|")
        except subprocess.CalledProcessError:
            print('Invalid python path: %s torch not installed.' %
                  python_bin_path)
            continue
        if not compile_args[0].startswith(_COMPAT_TORCH_VERSION):
            print('Invalid python path: %s compat torch version is %s'
                  ' got %s.' % (python_bin_path, _COMPAT_TORCH_VERSION,
                                compile_args[0]))
            continue
        # Write tools/python_bin_path.sh
        with open(real_config_path('PYTHON_BIN_PATH'), 'w') as f:
            f.write(python_bin_path)
        with open(real_config_path('TORCH_INSTALLED_PATH'), 'w') as f:
            f.write(compile_args[1])
        with open(real_config_path('COMPILE_FLAGS'), 'w') as f:
            for flag in compile_args[2:-1]:
                f.write("".join([flag, '\n']))
            f.write("".join(["-I", compile_args[-1], '\n']))
        break


def setup_ascend_sdk(env_path):
    """Get ascend install path."""
    ask_ascend_path = f'Specify the location of ascend sdk for debug on localhost or leave empty.' + \
        f'\n(You can make this quiet by set env [{_ASCEND_SDK_ENV}]): '
    custom_ascend_path = env_path
    while True:
        if not custom_ascend_path:
            ascend_path = get_input(ask_ascend_path)
        else:
            ascend_path = custom_ascend_path
            custom_ascend_path = None

        if not ascend_path:
            print(f"No ascend sdk path specified, skip setting up 'ASCEND_SDK_HEADERS_PATH'")
            return

        # Check if the path is valid
        if os.path.isdir(ascend_path) and os.access(ascend_path, os.X_OK):
            break
        if not os.path.exists(ascend_path):
            print('Invalid ascend path: %s cannot be found.' % ascend_path)

    with open(real_config_path('ASCEND_SDK_PATH'), 'w') as f:
        f.write(ascend_path)

    with open(real_config_path('env.sh'), 'w') as f:
        stub_libs = os.path.dirname(os.path.abspath(__file__)) + "/build/stubs"
        sdk_libs = f"{ascend_path}/lib"
        f.write(f"#!/bin/sh\n")
        f.write(f'export LD_LIBRARY_PATH={stub_libs}:{sdk_libs}')


def main():
    """Entry point for configuration"""
    env_snapshot = dict(os.environ)
    setup_python(env_snapshot.get(_PYTHON_BIN_PATH_ENV))
    if not env_snapshot.get(_NO_ASCEND_SDK) in ["1", "true", "True"]:
        setup_ascend_sdk(env_snapshot.get(_ASCEND_SDK_ENV))


if __name__ == '__main__':
    main()
