import re
import sys
import os
import types
import fcntl
import functools
import stat
from pathlib import Path
from typing import List
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from inductor_npu_ext import config


class StrRep:
    def __init__(self, value, str_value=None):
        self.value = value
        self.str_value = str_value if str_value else value

    def __str__(self):
        return self.str_value

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, value: object) -> bool:
        return self.value == value


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def is_kernel_need_stub(_):
    if config._debugging_on_cpu:
        return True
    return False


def _load_stub_modules(graph_name):
    need_stub = is_kernel_need_stub(graph_name)
    if need_stub:
        from inductor_npu_ext.common.revert_ascir import PyAutofuseStub, AscCompilerStub
        sys.modules["autofuse"] = types.ModuleType("autofuse")
        sys.modules["autofuse.pyautofuse"] = PyAutofuseStub()
        sys.modules["autofuse.compile_adapter"] = AscCompilerStub()
    try:
        yield
    finally:
        if need_stub:
            sys.modules.pop('autofuse', None)
            sys.modules.pop('autofuse.pyautofuse', None)
            sys.modules.pop('autofuse.compile_adapter', None)


@contextmanager
def load_compiler(graph_name):
    yield from _load_stub_modules(graph_name)


@contextmanager
def load_autofuser(graph_name):
    yield from _load_stub_modules(graph_name)


def validate_lib(filepath: str, *, change_permissions=False) -> None:
    if os.path.islink(filepath):
        raise PermissionError(f"{filepath} must not be a symbolic link")

    st = os.stat(filepath)

    # 必须是普通文件
    if not stat.S_ISREG(st.st_mode):
        raise PermissionError(f"{filepath} must be a regular file")

    # owner 必须是当前用户或 root
    current_uid = os.getuid()
    if st.st_uid not in (current_uid, 0):
        raise PermissionError(
            f"{filepath} must be owned by the current user or root. "
            f"Owned by UID {st.st_uid}, current UID {current_uid}"
        )

    actual_perms = st.st_mode & 0o777

    # group / other 不能有 w 位
    unsafe_bits = actual_perms & (stat.S_IWGRP | stat.S_IWOTH)
    if unsafe_bits:
        if change_permissions:
            os.chmod(filepath, actual_perms & ~(stat.S_IWGRP | stat.S_IWOTH))
        else:
            raise PermissionError(
                f"{filepath} must not be writable by group or others, "
                f"but got {oct(actual_perms)}"
            )


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def patch_fn(model, fn):
    orig_fn = getattr(model, fn)

    def decorator(f):
        @functools.wraps(orig_fn)
        def inner(*args, **kwargs):
            return f(*args, **kwargs, orig_fn=orig_fn)

        setattr(model, fn, inner)
        return inner

    return decorator


def get_node_meta(nodes: List[torch.fx.Node]):
    nodes = [nodes] if isinstance(nodes, torch.fx.Node) else nodes
    metas = []
    for n in nodes:
        if 'val' not in n.meta:
            continue
        metas.extend(pytree.tree_leaves(n.meta['val']))
    return metas
