"""Construct CompilerConfig configuration"""

__all__ = ["CompilerConfig"]

import os
from typing import Any, Optional
from datetime import datetime, timezone

import torch.distributed as dist

from npugraph_ex.configs._option_base import OptionValue, CallableValue, NpuBaseConfig, IntRangeValue, \
    MustExistedPathValue, DictOptionValue, IntListValue

INT64_MAX = 2 ** 63 - 1


def _timestamp():
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S%f")


class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.force_eager = OptionValue(False, [True, False])
        self.use_graph_pool = None
        self.reuse_graph_pool_in_same_fx = OptionValue(True, [True, False])
        self.capture_limit = IntRangeValue(64, 1, INT64_MAX)
        self.clone_input = OptionValue(True, [True, False])
        self.clone_output = OptionValue(False, [True, False])
        self.disable_static_kernel_compile_cache = OptionValue(False, [True, False])
        self.static_kernel_compile = OptionValue(False, [True, False])
        self.frozen_parameter = OptionValue(False, [True, False])
        self.remove_noop_ops = OptionValue(True, [True, False])
        self.remove_cat_ops = OptionValue(True, [True, False])
        self.inplace_pass = OptionValue(True, [True, False])
        self.input_inplace_pass = OptionValue(True, [True, False])
        self.pattern_fusion_pass = OptionValue(True, [True, False])
        self.post_grad_custom_pre_pass = CallableValue(None)
        self.post_grad_custom_post_pass = CallableValue(None)
        self.dump_tensor_data = OptionValue(False, [True, False])
        self.data_dump_stage = OptionValue('optimized', ['original', 'optimized'])
        self.data_dump_dir = MustExistedPathValue("./")
        self._vllm_aclnn_static_kernel_sym_index = IntRangeValue(0, 0, INT64_MAX)
        self._vllm_aclnn_static_kernel_sym_range = IntListValue(None)
        self.super_kernel_optimize = OptionValue(False, [True, False])
        self.super_kernel_optimize_options = DictOptionValue(None)
        self.super_kernel_debug_options = DictOptionValue(None)
        self.deadlock_check = OptionValue(False, [True, False])
        self.capture_error_mode = OptionValue("global", ["global", "thread_local", "relaxed"])
        self.mode = OptionValue("npugraph_ex", ["npugraph_ex"])

        super(CompilerConfig, self).__init__()
        self._fixed_attrs.append("post_grad_custom_pre_pass")
        self._fixed_attrs.append("post_grad_custom_post_pass")
        self._fixed_attrs.append("use_graph_pool")


    def data_dump_full_path(self):
        path = self.data_dump_dir.value
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            global_rank = dist.get_rank()
            path = os.path.join(path, f'worldsize{dist.get_world_size()}_global_rank{global_rank}')
        else:
            path = os.path.join(path, f'worldsize1_global_rank0')
        os.makedirs(path, exist_ok=True)
        return path

    def eager_data_dump_full_path(self, name: str, *, with_timestap=True):
        if not (self.force_eager and self.data_dump_stage):
            return None

        path = "." if self.data_dump_dir.value is None else self.data_dump_dir.value
        rank_id = dist.get_rank() if dist.is_initialized() else 0

        if with_timestap:
            return f"{path}/{name}_rank_{rank_id}_pid_{os.getpid()}_ts_{_timestamp()}.npy"
        else:
            return f"{path}/{name}_rank_{rank_id}_pid_{os.getpid()}.npy"


    def as_dict(self):
        local_option = {}
        # _DebugConfig
        local_option["force_eager"] = self.force_eager.value

        # _AclGraphDebugConfig
        local_option["reuse_graph_pool_in_same_fx"] = self.reuse_graph_pool_in_same_fx.value
        local_option["clone_output"] = self.clone_output.value
        # capture_limit must be str(int), so int(capture_limit) is always safe.
        local_option["capture_limit"] = self.capture_limit.value
        local_option["clone_input"] = self.clone_input.value
        local_option["deadlock_check"] = self.deadlock_check.value
        local_option["capture_error_mode"] = self.capture_error_mode.value

        # _AclGraphExperimentalConfig
        # The parameters "static_kernel_compile"and "_aclnn_static_shape_kernel_build_dir" will be ignored
        # if the "static_kernel_compile" parameter is set to False("0").
        # The value of static_kernel_compile.value will be "0" by default.
        if self.static_kernel_compile:
            local_option["static_kernel_compile"] = self.static_kernel_compile.value

            # "_vllm_aclnn_static_kernel_sym_range" is used to specify the range of symbols
            # that need to compile static shape kernel for the current FX graph.
            # The default value is None, which means that all symbol values need to trigger static compilation.
            local_option["_vllm_aclnn_static_kernel_sym_range"] = self._vllm_aclnn_static_kernel_sym_range.value

            # "_aclnn_static_shape_kernel_sym_index" is used to specify the index of symbols
            # that need to compile static shape kernel for the current FX graph.
            # The default value is 0, which means that the first symbol needs to be checked
            # whether the value is within the specified range by "_vllm_aclnn_static_kernel_sym_range".
            local_option["_vllm_aclnn_static_kernel_sym_index"] = self._vllm_aclnn_static_kernel_sym_index.value

        if self.disable_static_kernel_compile_cache.value:
            local_option["disable_static_kernel_compile_cache"] = self.disable_static_kernel_compile_cache.value

        if self.super_kernel_optimize.value:
            local_option["super_kernel_optimize"] = self.super_kernel_optimize.value
            if self.super_kernel_optimize_options.value is not None:
                local_option["super_kernel_optimize_options"] = self.super_kernel_optimize_options.value
            if self.super_kernel_debug_options.value is not None:
                local_option["super_kernel_debug_options"] = self.super_kernel_debug_options.value
        # _ExperimentalConfig
        local_option["remove_noop_ops"] = self.remove_noop_ops.value
        local_option["pattern_fusion_pass"] = self.pattern_fusion_pass.value
        local_option["frozen_parameter"] = self.frozen_parameter.value
        # _DataDumpConfig
        if self.dump_tensor_data:
            local_option['dump_tensor_data'] = self.dump_tensor_data.value
            local_option['data_dump_path'] = self.data_dump_full_path()
            local_option['data_dump_stage'] = self.data_dump_stage.value

        return local_option, {}


def _get_all_leaf_properties(obj: Any, prefix: str = ""):
    stack = [(prefix, obj)]
    leaves = []

    while stack:
        current_prefix, current_obj = stack.pop()

        try:
            attrs = vars(current_obj).copy()
        except TypeError:
            leaves.append((current_prefix.rstrip("."), current_obj))
            continue

        if not attrs:
            leaves.append((current_prefix.rstrip("."), current_obj))
            continue

        for key, value in attrs.items():
            current_path = f"{current_prefix}{key}"
            try:
                child_attrs = vars(value).copy()
            except TypeError:
                child_attrs = {}

            if child_attrs:
                stack.append((current_path + ".", value))
                continue
            if value is None:
                continue
            leaf_value = value.value if hasattr(value, "value") else value
            leaves.append((current_path, leaf_value))
    return leaves


def _process_kwargs_options(config, kwargs):
    """
    Processes the "options" parameter to config.


    Args:
        config: Configuration object to be updated with valid option settings
        kwargs: Parameter dictionary that may contain the "options" entry to be processed


    Raises:
        ValueError: Triggered when the "options" entry contains identifiers not specified
                    in the predefined mapping rules
    """

    # Process target parameter
    if "options" in kwargs:
        options = kwargs["options"]
        for option in options:
            setattr(config, option, options[option])
