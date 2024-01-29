import os
from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig
from torchair.configs.utils import check_file


class FusionConfig(NpuBaseConfig):
    """Config for op fusion"""

    def __init__(self):
        self.fusion_switch_file = OptionValue(None, check_file, "fusion_config.fusion_switch_file")

        super(FusionConfig, self).__init__()

    def as_dict(self):
        fusion_option = {}
        if self.fusion_switch_file.value is not None:
            fusion_option['ge.fusionSwitchFile'] = self.fusion_switch_file.value
        return {}, fusion_option
