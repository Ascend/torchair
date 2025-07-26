__all__ = []

from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig


class _AclGraphConfig:
    """
    Config for AclGraph option
    """

    def __init__(self) -> None:
        self.use_custom_pool = None

        super(_AclGraphConfig, self).__init__()
