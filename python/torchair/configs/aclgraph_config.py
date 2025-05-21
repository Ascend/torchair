__all__ = []

from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig


class _AclgraphConfig(NpuBaseConfig):
    """Config for aclgraph option"""

    def __init__(self):
        self.disable_reinplace_inplaceable_ops_pass = OptionValue(False, [True, False])
        self.disable_reinplace_input_mutated_ops_pass = OptionValue(False, [True, False])

        super(_AclgraphConfig, self).__init__()

