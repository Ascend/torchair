from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


def _counter():
    i = 0
    while True:
        yield i
        i += 1

_uuid = _counter()

class _Dump(NpuBaseConfig):
    def __init__(self):
        self.type = OptionValue(None, ["txt", "pbtxt"])
        self.path = OptionValue(None, None)
        self.prefix = OptionValue(None, None)

        super(_Dump, self).__init__()

    def full_path(self, name: str):
        if self.path.value is None and self.prefix.value is None and self.type.value is None:
            return None

        path = "." if self.path.value is None else self.path.value
        prefix = "" if self.prefix.value is None else self.prefix.value + "_"
        suffix = next(_uuid)
        suffix = ("_" + str(suffix)) if suffix > 0 else ""
        type = "txt" if self.type.value is None else self.type.value

        return f"{path}/{prefix}{name}{suffix}.{type}"


class DebugConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.graph_dump = _Dump()

        super(DebugConfig, self).__init__()

    def as_dict(self):
        return {}
