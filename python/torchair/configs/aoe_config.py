from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class AoeConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.aoe_mode = OptionValue(None, ["1", "2", "4"])
        self.work_path = OptionValue("./", None)
        self.aoe_config_file = OptionValue(None, None)

        super(AoeConfig, self).__init__()

    def as_dict(self):
        if self.aoe_mode.value is None:
            return {}
        return super().as_dict()
