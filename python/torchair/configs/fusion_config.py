import os
from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class FusionConfig(NpuBaseConfig):
    """Config for op fusion"""

    def __init__(self):
        self.fusion_switch_file = OptionValue(None, None)

        super(FusionConfig, self).__init__()

    def as_dict(self):
        fusion_option = {}
        if self.fusion_switch_file.value is not None:
            if not (os.path.exists(self.fusion_switch_file.value) and os.path.isfile(self.fusion_switch_file.value)):
                raise FileNotFoundError("fusion_config.fusion_switch_file " + self.fusion_switch_file.value +
                                        " is not found or is not a file! Please change!")
            fusion_option['ge.fusionSwitchFile'] = self.fusion_switch_file.value
        return {}, fusion_option
