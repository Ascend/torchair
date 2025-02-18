__all__ = []

from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig


class _InferenceConfig(NpuBaseConfig):
    """Config for inference"""

    def __init__(self):
        self.dynamic_gears_merge_policy = OptionValue("zip", ["zip", "product"])

        super(_InferenceConfig, self).__init__()

    def as_dict(self):
        return {}, {}

