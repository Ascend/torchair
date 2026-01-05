__all__ = []

from typing import Optional

from torchair.configs._option_base import OptionValue, MustExistedPathValue, FileValue
from torchair.configs._option_base import NpuBaseConfig


class _AoeConfig(NpuBaseConfig):
    """Config for aoe function"""

    def __init__(self):
        self.aoe_mode = OptionValue(None, ["2"])
        self.work_path = MustExistedPathValue("./")
        self.aoe_config_file = FileValue(None)

        super(_AoeConfig, self).__init__()

    def as_dict(self, mode: Optional[str] = "max-autotune"):
        if self.aoe_mode.value is None:
            return {}, {}
        return super().as_dict()
