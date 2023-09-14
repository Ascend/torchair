from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class AotConfig(NpuBaseConfig):
    """Config for custom aot functions"""

    def __init__(self):
        self.enable_joint_graph = OptionValue(False, [True, False])
        self.output_loss_index = OptionValue(0, None)

        super(AotConfig, self).__init__()