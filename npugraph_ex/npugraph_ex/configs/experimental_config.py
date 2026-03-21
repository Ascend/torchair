__all__ = []


from npugraph_ex.configs._option_base import OptionValue, IntRangeValue, StrOptionValue, IntListValue, DictOptionValue
from npugraph_ex.configs._option_base import NpuBaseConfig

INT64_MAX = 2 ** 63 - 1


class _ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.keep_inference_input_mutations = OptionValue(True, [True, False])
        self.frozen_parameter = OptionValue(False, [True, False])
        self.npu_fx_pass = OptionValue(False, [True, False])
        self.remove_noop_ops = OptionValue(True, [True, False])
        self.pattern_fusion_pass = OptionValue(True, [True, False])
        self.aclgraph = _AclGraphExperimentalConfig()

        super(_ExperimentalConfig, self).__init__()

    def as_dict(self):
        global_experiment_option = {}
        local_experiment_option = {"remove_noop_ops": self.remove_noop_ops.value,
                                   "pattern_fusion_pass": self.pattern_fusion_pass.value,
                                   "frozen_parameter": self.frozen_parameter.value}
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
        self._super_kernel_optimize = OptionValue(False, [True, False])
        self._super_kernel_optimize_options = DictOptionValue(None)
        self._super_kernel_debug_options = DictOptionValue(None)

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

        if self._super_kernel_optimize.value == "1":
            local_aclgraph_experimental_options["_super_kernel_optimize"] = self._super_kernel_optimize.value
            if self._super_kernel_optimize_options.value is not None:
                local_aclgraph_experimental_options[
                    "_super_kernel_optimize_options"] = self._super_kernel_optimize_options.value
            if self._super_kernel_debug_options.value is not None:
                local_aclgraph_experimental_options[
                    "_super_kernel_debug_options"] = self._super_kernel_debug_options.value

        return local_aclgraph_experimental_options, global_aclgraph_experimental_options

