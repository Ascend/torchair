__all__ = []

from torchair.configs._option_base import OptionValue, IntRangeValue
from torchair.configs._option_base import NpuBaseConfig


class _ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.cc_parallel_enable = OptionValue(False, [False, True])
        self.keep_inference_input_mutations = OptionValue(True, [True, False])
        self.memory_efficiency = OptionValue(False, [True, False])
        self.frozen_parameter = OptionValue(False, [True, False])
        self.static_model_ops_lower_limit = IntRangeValue(None, -1, 9223372036854775807)
        self.jit_compile = OptionValue("auto", ["auto"])
        self.npu_fx_pass = OptionValue(False, [True, False])
        self.aot_config_enable_joint_graph = OptionValue(False, [True, False])
        self.aot_config_output_loss_index = OptionValue(0, None)
        self.topology_sorting_strategy = OptionValue("DFS", ["BFS", "DFS", "RDFS", "StableRDFS"])
        self.enable_ref_data = OptionValue(False, [True, False])
        self.enable_view_optimize = OptionValue(True, [True, False])
        self.tiling_schedule_optimize = OptionValue(False, [True, False])

        super(_ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {}
        sorting_strategy_dict = {"BFS": "0", "DFS": "1", "RDFS": "2", "StableRDFS": "3"}

        global_experiment_option["ge.exec.enableEngineParallel"] = "1" if self.cc_parallel_enable else "0"
        global_experiment_option["ge.tiling_schedule_optimize"] = "1" if self.tiling_schedule_optimize else "0"
        local_experiment_option["ge.featureBaseRefreshable"] = "1" if self.memory_efficiency else "0"
        local_experiment_option["ge.topoSortingMode"] = sorting_strategy_dict[self.topology_sorting_strategy.value]
        if self.jit_compile.value == "auto":
            local_experiment_option["ge.jit_compile"] = "2"
        if self.static_model_ops_lower_limit.value is not None:
            local_experiment_option["ge.exec.static_model_ops_lower_limit"] = \
                str(self.static_model_ops_lower_limit.value)
        return local_experiment_option, global_experiment_option
    
