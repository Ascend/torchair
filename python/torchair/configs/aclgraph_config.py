__all__ = []

from torchair.configs._option_base import OptionValue, MustExistedPathValue
from torchair.configs._option_base import NpuBaseConfig


class _AclGraphConfig:
    """
    Config for AclGraph option
    """

    def __init__(self) -> None:
        self.use_custom_pool = None
        self.kernel_aot_optimization = OptionValue(False, [True, False])
        self.kernel_aot_optimization_build_dir = None

        super(_AclGraphConfig, self).__init__()
