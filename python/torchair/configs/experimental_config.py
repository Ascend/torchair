__all__ = []

from torchair.configs._option_base import OptionValue, IntRangeValue, StrOptionValue
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
        self.remove_noop_ops = OptionValue(True, [True, False])
        self.tiling_schedule_optimize = OptionValue(False, [True, False])
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

        return local_aclgraph_experimental_options, global_aclgraph_experimental_options

