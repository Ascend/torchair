from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.cc_parallel_enable = OptionValue(False, [False, True])

        super(ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        if self.cc_parallel_enable.value is True:
            global_experiment_option["ge.enableEngineParallel"] = "1"
        return {}, global_experiment_option
