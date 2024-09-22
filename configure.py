"""Basic configurations for building torchair"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import subprocess
import sys
import warnings

_COMPAT_TORCH_VERSION = "2.1"
_PYTHON_BIN_PATH_ENV = "TARGET_PYTHON_PATH"
_ASCEND_SDK_ENV = "ASCEND_SDK_PATH"
_NO_ASCEND_SDK = "NO_ASCEND_SDK"


class PathManager:
    MAX_PATH_LENGTH = 4096
    MAX_FILE_NAME_LENGTH = 255
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_writeable(cls, path):
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.W_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def create_file_safety(cls, path: str):
        msg = f"Failed to create file: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg) from err


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


def get_torch_version_num(version):
    nums = re.findall(r'\d+', version)
    if len(nums) < 3:
        print("Parse torch version failure, version:%s" % version)
        return ''
    return "{}{:0>2d}{:0>2d}".format(nums[0], int(nums[1]), int(nums[2]))


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
        if not compile_args[0] >= _COMPAT_TORCH_VERSION:
            print('Invalid python path: %s expect torch version >= %s'
                  ' got %s.' % (python_bin_path, _COMPAT_TORCH_VERSION,
                                compile_args[0]))
            continue
        for path in ['PYTHON_BIN_PATH', 'TORCH_INSTALLED_PATH', 'COMPILE_FLAGS']:
            real_path = os.path.abspath(real_config_path(path))
            PathManager.create_file_safety(real_path)
            PathManager.check_directory_path_writeable(real_path)
        # Write tools/python_bin_path.sh
        with open(real_config_path('PYTHON_BIN_PATH'), 'w') as f:
            f.write(python_bin_path)
        with open(real_config_path('TORCH_INSTALLED_PATH'), 'w') as f:
            f.write(compile_args[1])
        with open(real_config_path('COMPILE_FLAGS'), 'w') as f:
            for flag in compile_args[2:-1]:
                f.write("".join([flag, '\n']))
            f.write("".join(["-I", compile_args[-1], '\n']))
        torch_version = get_torch_version_num(compile_args[0])
        if len(torch_version) > 0:
            with open(real_config_path('TORCH_VERSION'), 'w') as f:
                f.write(torch_version)
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
            print(f"No ascend sdk path specified")
            return

        # Check if the path is valid
        if os.path.isdir(ascend_path) and os.access(ascend_path, os.X_OK):
            break
        if not os.path.exists(ascend_path):
            print('Invalid ascend path: %s cannot be found.' % ascend_path)

    for path in ['ASCEND_SDK_PATH', 'env.sh']:
        real_path = os.path.abspath(real_config_path(path))
        PathManager.create_file_safety(real_path)
        PathManager.check_directory_path_writeable(real_path)

    with open(real_config_path('ASCEND_SDK_PATH'), 'w') as f:
        f.write(ascend_path)

    with open(real_config_path('env.sh'), 'w') as f:
        stub_libs = os.path.dirname(os.path.abspath(__file__)) + "/build/stubs"
        sdk_libs = f"{ascend_path}/lib:{ascend_path}/lib64"
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
