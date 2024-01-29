from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig
from torchair.configs.utils import check_dir_path, check_file


class AoeConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.aoe_mode = OptionValue(None, ["2"])
        self.work_path = OptionValue("./", check_dir_path, "aoe_config.work_path")
        self.aoe_config_file = OptionValue(None, check_file, "aoe_config.aoe_config_file")

        super(AoeConfig, self).__init__()

    def as_dict(self):
        if self.aoe_mode.value is None:
            return {}, {}
        return super().as_dict()
