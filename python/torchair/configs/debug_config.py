__all__ = []

from datetime import datetime, timezone
import os
import torch.distributed as dist
from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig


def _timestamp():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")


class _DebugBase(NpuBaseConfig):
    def __init__(self):
        self._path = OptionValue(None, None)

        super(_DebugBase, self).__init__()

    @property
    def enabled(self):
        return self.type.value is not None

    def full_path(self, name: str, *, with_timestap=True):
        if not self.enabled:
            return None

        path = "." if self._path.value is None else self._path.value
        rank_id = dist.get_rank() if dist.is_initialized() else 0

        if with_timestap:
            return f"{path}/{name}_rank_{rank_id}_pid_{os.getpid()}_ts_{_timestamp()}.{self.type.value}"
        else:
            return f"{path}/{name}_rank_{rank_id}_pid_{os.getpid()}.{self.type.value}"


class _Dump(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["txt", "pbtxt", "py"])
        super(_Dump, self).__init__()
        self._fixed_attrs.append('path')

    @property
    def path(self):
        return self._path.value

    @path.setter
    def path(self, value):
        self._path.value = os.path.realpath(value)


class _DataDump(_DebugBase):
    def __init__(self):
        self.filter = None
        self.type = OptionValue(None, ["npy"])
        super(_DataDump, self).__init__()
        self._fixed_attrs.append("filter")
        self._fixed_attrs.append('path')

    @property
    def path(self):
        return self._path.value

    @path.setter
    def path(self, value):
        self._path.value = os.path.realpath(value)


class _FxSummary(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["csv"])
        self.skip_compile = OptionValue(True, [True, False])
        super(_FxSummary, self).__init__()


class _AclGraphDebugConfig(NpuBaseConfig):
    """Config for aclgraph debug option"""

    def __init__(self) -> None:
        self.disable_reinplace_inplaceable_ops_pass = OptionValue(False, [True, False])
        self.disable_reinplace_input_mutated_ops_pass = OptionValue(False, [True, False])
        self.disable_mempool_reuse_in_same_fx = OptionValue(False, [True, False])
        self.enable_output_clone = OptionValue(False, [True, False])

        super(_AclGraphDebugConfig, self).__init__()

    def as_dict(self):
        local_option = {}
        local_option["enable_output_clone"] = self.enable_output_clone.value

        if self.disable_mempool_reuse_in_same_fx.value is not None:
            local_option["disable_mempool_reuse_in_same_fx"] = self.disable_mempool_reuse_in_same_fx.value
        return local_option


class _DebugConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.graph_dump = _Dump()
        self.data_dump = _DataDump()
        self.fx_summary = _FxSummary()
        self.run_eagerly = OptionValue(False, [True, False])
        self.aclgraph = _AclGraphDebugConfig()

        super(_DebugConfig, self).__init__()

    def as_dict(self):
        return {}, {}
