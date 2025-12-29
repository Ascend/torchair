__all__ = []

from torchair.configs._option_base import OptionValue, IntRangeValue, StrOptionValue, IntListValue
from torchair.configs._option_base import NpuBaseConfig

INT64_MAX = 2 ** 63 - 1


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
        self.remove_noop_ops = OptionValue(True, [True, False])
        self.tiling_schedule_optimize = OptionValue(False, [True, False])
        self.pattern_fusion_pass = OptionValue(True, [True, False])
        self.aclgraph = _AclGraphExperimentalConfig()

        super(_ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {}
        sorting_strategy_dict = {"BFS": "0", "DFS": "1", "RDFS": "2", "StableRDFS": "3"}

        global_experiment_option["ge.exec.enableEngineParallel"] = "1" if self.cc_parallel_enable else "0"
        global_experiment_option["ge.tiling_schedule_optimize"] = "1" if self.tiling_schedule_optimize else "0"
        local_experiment_option["remove_noop_ops"] = self.remove_noop_ops.value
        local_experiment_option["ge.featureBaseRefreshable"] = "1" if self.memory_efficiency else "0"
        local_experiment_option["ge.topoSortingMode"] = sorting_strategy_dict[self.topology_sorting_strategy.value]
        local_experiment_option["pattern_fusion_pass"] = self.pattern_fusion_pass.value
        if self.jit_compile.value == "auto":
            local_experiment_option["ge.jit_compile"] = "2"
        if self.static_model_ops_lower_limit.value is not None:
            local_experiment_option["ge.exec.static_model_ops_lower_limit"] = \
                str(self.static_model_ops_lower_limit.value)
        local_aclgraph_experimental_options, global_aclgraph_experimental_options = self.aclgraph.as_dict()
        local_experiment_option.update(local_aclgraph_experimental_options)
        global_experiment_option.update(global_aclgraph_experimental_options)
        return local_experiment_option, global_experiment_option


class _AclGraphExperimentalConfig(NpuBaseConfig):
    def __init__(self):
        self._aclnn_static_shape_kernel = OptionValue(False, [True, False])
        self._aclnn_static_shape_kernel_build_dir = StrOptionValue()
        self._aclnn_static_shape_kernel_sym_value_range = IntListValue(None)
        self._aclnn_static_shape_kernel_sym_index = IntRangeValue(0, 0, INT64_MAX)

        super(_AclGraphExperimentalConfig, self).__init__()

    def as_dict(self):
        global_aclgraph_experimental_options = {}
        local_aclgraph_experimental_options = {}

        # The parameters "_aclnn_static_shape_kernel"and "_aclnn_static_shape_kernel_build_dir" will be ignored
        # if the "_aclnn_static_shape_kernel" parameter is set to False("0").
        # The value of _aclnn_static_shape_kernel.value will be "0" by default.
        if self._aclnn_static_shape_kernel.value == "1":
            local_aclgraph_experimental_options["_aclnn_static_shape_kernel"] = self._aclnn_static_shape_kernel.value
            # Explicitly setting "_aclnn_static_shape_kernel_build_dir" to None will cause runtime errors
            # because the underlying ge::AscendString type cannot handle None values.
            # Since "_aclnn_static_shape_kernel_build_dir" is used directly,
            # setting its default value to an empty string ("") in the as_dict function will not cause any issues.
            local_aclgraph_experimental_options["_aclnn_static_shape_kernel_build_dir"] = \
                self._aclnn_static_shape_kernel_build_dir.value if self._aclnn_static_shape_kernel_build_dir.value\
                                                                   is not None else ""

            # "_aclnn_static_shape_kernel_sym_value_range" is used to specify the range of symbols
            # that need to compile static shape kernel for the current FX graph.
            # The default value is None, which means that all symbol values need to trigger static compilation.
            local_aclgraph_experimental_options["_aclnn_static_shape_kernel_sym_value_range"] = \
                self._aclnn_static_shape_kernel_sym_value_range.value

            # "_aclnn_static_shape_kernel_sym_index" is used to specify the index of symbols
            # that need to compile static shape kernel for the current FX graph.
            # The default value is 0, which means that the first symbol needs to be checked
            # whether the value is within the specified range by "_aclnn_static_shape_kernel_sym_value_range".
            local_aclgraph_experimental_options["_aclnn_static_shape_kernel_sym_index"] = \
                self._aclnn_static_shape_kernel_sym_index.value

        return local_aclgraph_experimental_options, global_aclgraph_experimental_options

