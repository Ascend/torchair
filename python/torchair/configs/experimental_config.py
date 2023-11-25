from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.cc_parallel_enable = OptionValue(False, [False, True])
        self.keep_inference_input_mutations = OptionValue(False, [True, False])
        self.memory_efficiency = OptionValue(False, [True, False])

        super(ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {}
        if self.cc_parallel_enable:
            global_experiment_option["ge.exec.enableEngineParallel"] = "1"
        if self.memory_efficiency:
            local_experiment_option["ge.featureBaseRefreshable"] = "1"
        return local_experiment_option, global_experiment_option
