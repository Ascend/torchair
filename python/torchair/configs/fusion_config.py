import os
from torchair.configs.option_base import FileValue
from torchair.configs.option_base import NpuBaseConfig


class FusionConfig(NpuBaseConfig):
    """Config for op fusion"""

    def __init__(self):
        self.fusion_switch_file = FileValue(None)

        super(FusionConfig, self).__init__()

    def as_dict(self):
        fusion_option = {}
        if self.fusion_switch_file.value is not None:
            fusion_option['ge.fusionSwitchFile'] = self.fusion_switch_file.value
        return {}, fusion_option
