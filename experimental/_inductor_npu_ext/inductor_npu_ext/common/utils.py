import re
import sys
import os
import types
import fcntl

from pathlib import Path
from contextlib import contextmanager
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
    if config._debugging_host_only:
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


def _validate_permissions(filepath: str, perms: int, *, change_permissions=False) -> None:
    if os.path.islink(filepath):
        raise PermissionError(f"{filepath} must not be a symbolic link")

    st = os.stat(filepath)

    current_uid = os.getuid()
    if st.st_uid != current_uid:
        raise PermissionError(
            f"{filepath} must be owned by the current user. "
            f"Owned by UID {st.st_uid}, current UID {current_uid}"
        )

    actual_perms = st.st_mode & 0o777
    if actual_perms != perms:
        if change_permissions:
            os.chmod(filepath, perms)
        else:
            raise PermissionError(
                f"{filepath} permissions must be {oct(perms)}, "
                f"but got {oct(actual_perms)}"
            )


def validate_file(filepath: str, *, change_permissions=False) -> None:
    _validate_permissions(filepath, 0o644, change_permissions=change_permissions)


def validate_lib(filepath: str, *, change_permissions=False) -> None:
    _validate_permissions(filepath, 0o755, change_permissions=change_permissions)


@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
