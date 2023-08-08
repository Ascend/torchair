from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig
from datetime import datetime


def _timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


class _DebugBase(NpuBaseConfig):
    def __init__(self):
        self.path = OptionValue(None, None)

        super(_DebugBase, self).__init__()

    @property
    def enabled(self):
        return self.type.value is not None

    def full_path(self, name: str):
        if not self.enabled:
            return None

        path = "." if self.path.value is None else self.path.value

        return f"{path}/{name}_{_timestamp()}.{self.type.value}"


class _Dump(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["txt", "pbtxt", "py"])
        super(_Dump, self).__init__()


class _FxSummary(_DebugBase):
    def __init__(self):
        self.type = OptionValue(None, ["csv"])
        self.skip_compile = OptionValue(True, [True, False])
        super(_FxSummary, self).__init__()


class DebugConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.graph_dump = _Dump()
        self.fx_summary = _FxSummary()

        super(DebugConfig, self).__init__()

    def as_dict(self):
        return {}
