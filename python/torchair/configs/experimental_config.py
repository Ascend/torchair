from torchair.configs.option_base import OptionValue, IntRangeValue
from torchair.configs.option_base import NpuBaseConfig


class ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.cc_parallel_enable = OptionValue(False, [False, True])
        self.keep_inference_input_mutations = OptionValue(True, [True, False])
        self.memory_efficiency = OptionValue(False, [True, False])
        self.separate_atomic_clean = OptionValue(True, [True, False])
        self.frozen_parameter = OptionValue(False, [True, False])
        self.static_model_ops_lower_limit = IntRangeValue(None, -1, 9223372036854775807)
        self.jit_compile = OptionValue("auto", ["auto"])
        self.npu_fx_pass = OptionValue(False, [True, False])
        self.aot_config_enable_joint_graph = OptionValue(False, [True, False])
        self.aot_config_output_loss_index = OptionValue(0, None)
        self.enable_single_stream = OptionValue(False, [True, False])

        super(ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {}

        global_experiment_option["ge.exec.enableEngineParallel"] = "1" if self.cc_parallel_enable else "0"
        global_experiment_option["ge.enableSingleStream"] = "true" if self.enable_single_stream else "false"
        local_experiment_option["ge.featureBaseRefreshable"] = "1" if self.memory_efficiency else "0"
        local_experiment_option["ge.exec.atomicCleanPolicy"] = "1" if self.separate_atomic_clean else "0"
        if self.jit_compile.value == "auto":
            local_experiment_option["ge.jit_compile"] = "2"
        if self.static_model_ops_lower_limit.value is not None:
            local_experiment_option["ge.exec.static_model_ops_lower_limit"] = \
                str(self.static_model_ops_lower_limit.value)
        return local_experiment_option, global_experiment_option
    
