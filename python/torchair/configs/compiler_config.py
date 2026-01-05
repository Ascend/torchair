"""Construct CompilerConfig configuration"""

__all__ = ["CompilerConfig"]

import warnings
from typing import Any, Optional
from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import CallableValue
from torchair.configs._option_base import DeprecatedValue
from torchair.configs._option_base import NpuBaseConfig
from torchair.configs.aoe_config import _AoeConfig
from torchair.configs.export_config import _ExportConfig
from torchair.configs.debug_config import _DebugConfig
from torchair.configs.dump_config import _DataDumpConfig
from torchair.configs.fusion_config import _FusionConfig
from torchair.configs.inference_config import _InferenceConfig
from torchair.configs.experimental_config import _ExperimentalConfig
from torchair.configs.ge_config import _GEConfig
from torchair.configs.aclgraph_config import _AclGraphConfig
from torchair.configs.npugraphex_config import _NpuGraphExConfig
from torchair.core.utils import logger


class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.debug = _DebugConfig()

        self.aoe_config = _AoeConfig()
        self.export = _ExportConfig()
        self.dump_config = _DataDumpConfig()
        self.fusion_config = _FusionConfig()
        self.experimental_config = _ExperimentalConfig()
        self.inference_config = _InferenceConfig()
        self.ge_config = _GEConfig()
        self.aclgraph_config = _AclGraphConfig()
        self.npugraphex_config = _NpuGraphExConfig()
        self.mode = OptionValue("max-autotune", ["max-autotune", "reduce-overhead"])
        self.post_grad_custom_pre_pass = CallableValue(None)
        self.post_grad_custom_post_pass = CallableValue(None)

        super(CompilerConfig, self).__init__()
        self._fixed_attrs.append("post_grad_custom_pre_pass")
        self._fixed_attrs.append("post_grad_custom_post_pass")

    def __eq__(self, other):
        if not isinstance(other, CompilerConfig):
            return False
        self_dict = dict(_get_all_leaf_properties(self))
        other_dict = dict(_get_all_leaf_properties(other))
        return self_dict == other_dict

    def as_dict(self):
        local_options, global_options = super().as_dict(self.mode.value)
        if "post_grad_custom_pre_pass" in local_options.keys():
            local_options.pop("post_grad_custom_pre_pass")
        if "post_grad_custom_post_pass" in local_options.keys():
            local_options.pop("post_grad_custom_post_pass")
        return local_options, global_options


unsupport_geconfig_list = [("debug.aclgraph.disable_reinplace_input_mutated_ops_pass", [True]),
    ("experimental_config.aclgraph._aclnn_static_shape_kernel", [True]),
    ("experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir", []),
    ("debug.aclgraph.clone_input", [False]),
    ("debug.aclgraph.disable_reinplace_inplaceable_ops_pass", [True])]
unsupport_aclgraphconfig_list = [("inference_config.dynamic_gears_merge_policy", ["product"]), \
    ("debug.fx_summary.type", ["csv"]), ("dump_config.enable_dump", [True]), \
    ("ge_config.export_compile_stat", ["1", "0"]), \
    ("export.experimental.auto_atc_config_generated", [True]), \
    ("export.experimental.enable_record_nn_module_stack", [True]), \
    ("ge_config.enable_single_stream", [True]), ("ge_config.oo_level", ["O1"]), \
    ("ge_config.oo_constant_folding", [True, False]), ("ge_config.oo_dead_code_elimination", [True, False]), \
    ("experimental_config.topology_sorting_strategy", ["BFS", "RDFS", "StableRDFS"]), \
    ("experimental_config.cc_parallel_enable", [True]), \
    ("experimental_config.enable_ref_data", [True]), \
    ("experimental_config.tiling_schedule_optimize", [True]), \
    ("experimental_config.enable_view_optimize", [False, True]), ("fusion_config.fusion_switch_file", []), \
    ("experimental_config.static_model_ops_lower_limit", []), ("ge_config.aicore_num", []), \
    ("ge_config.optimization_switch", [])]


def _check_config_support(config: Any):
    warnings.filterwarnings("once", category=UserWarning)
    config_dict = dict(_get_all_leaf_properties(config))
    warn_config = []
    if config.mode.value == "max-autotune":
        config_list = unsupport_geconfig_list
    else:
        config_list = unsupport_aclgraphconfig_list

    for config_arg in config_list:
        key_raw = config_arg[0]
        key_with_value = key_raw + "._value"
        if key_raw in config_dict or key_with_value in config_dict:
            warn_config = _get_warn_config(warn_config, config_arg, config_dict)

    if warn_config:
        mode_specific = "max-autotune" if config.mode.value == "max-autotune" else "reduce-overhead"
        additional = (
            ""
            if mode_specific == "max-autotune"
            else ", set_dim_gears, dynamo_export, scope, npu_print"
        )
        warnings.warn(
            f"The following torchair config or properties may not take effect or report "
            f"error in {mode_specific} mode: {', '.join(warn_config)}{additional}",
            UserWarning
        )


def _get_warn_config(warn_config, config_arg, config_dict):
    key_raw = config_arg[0]
    key_with_value = key_raw + "._value"
    value = config_dict.get(key_raw) or config_dict.get(key_with_value)
    if value is not None:
        if not config_arg[1] or value in config_arg[1]:
            warn_config.append(f"config.{key_raw}:{value}")
    return warn_config


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
