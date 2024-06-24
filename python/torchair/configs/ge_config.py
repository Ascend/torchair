from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig

__all__ = []


class _GEConfig(NpuBaseConfig):
    """Config for ge option"""

    def __init__(self):
        self.enable_single_stream = OptionValue(False, [True, False])

        super(_GEConfig, self).__init__()

    def as_dict(self):
        global_option = {}
        local_option = {}
        global_option["ge.enableSingleStream"] = "true" if self.enable_single_stream else "false"
        return local_option, global_option