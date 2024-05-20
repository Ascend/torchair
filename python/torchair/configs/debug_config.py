from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig
from datetime import datetime

__all__ = []


def _timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


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

        if with_timestap:
            return f"{path}/{name}_{_timestamp()}.{self.type.value}"
        else:
            return f"{path}/{name}.{self.type.value}"


class _Dump(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["txt", "pbtxt", "py"])
        super(_Dump, self).__init__()


class _DataDump(_DebugBase):
    def __init__(self):
        self.filter = None
        self.type = OptionValue(None, ["npy"])
        super(_DataDump, self).__init__()
        self._fixed_attrs.append("filter")


class _FxSummary(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["csv"])
        self.skip_compile = OptionValue(True, [True, False])
        super(_FxSummary, self).__init__()


class _DebugConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.graph_dump = _Dump()
        self.data_dump = _DataDump()
        self.fx_summary = _FxSummary()

        super(_DebugConfig, self).__init__()

    def as_dict(self):
        return {}, {}
