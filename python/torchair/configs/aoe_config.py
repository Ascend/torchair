from torchair.configs._option_base import OptionValue, MustExistedPathValue, FileValue
from torchair.configs._option_base import NpuBaseConfig

__all__ = []


class _AoeConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.aoe_mode = OptionValue(None, ["2"])
        self.work_path = MustExistedPathValue("./")
        self.aoe_config_file = FileValue(None)

        super(_AoeConfig, self).__init__()

    def as_dict(self):
        if self.aoe_mode.value is None:
            return {}, {}
        return super().as_dict()
