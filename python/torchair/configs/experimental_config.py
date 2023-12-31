from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.cc_parallel_enable = OptionValue(False, [False, True])
        self.keep_inference_input_mutations = OptionValue(True, [True, False])
        self.memory_efficiency = OptionValue(False, [True, False])
        self.separate_atomic_clean = OptionValue(True, [True, False])

        super(ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {}

        global_experiment_option["ge.exec.enableEngineParallel"] = "1" if self.cc_parallel_enable else "0"
        local_experiment_option["ge.featureBaseRefreshable"] = "1" if self.memory_efficiency else "0"
        local_experiment_option["ge.exec.atomicCleanPolicy"] = "1" if self.separate_atomic_clean else "0"

        return local_experiment_option, global_experiment_option
