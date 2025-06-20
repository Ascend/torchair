"""Construct CompilerConfig configuration"""

__all__ = ["CompilerConfig"]

import warnings
from typing import Any
from torchair.configs._option_base import OptionValue
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
        self.mode = OptionValue("max-autotune", ["max-autotune", "reduce-overhead"])

        super(CompilerConfig, self).__init__()


    def __eq__(self, other):
        if not isinstance(other, CompilerConfig):
            return False
        self_dict = dict(_get_all_leaf_properties(self))
        other_dict = dict(_get_all_leaf_properties(other))
        return self_dict == other_dict


unsupport_geconfig_list = [("debug.aclgraph.disable_reinplace_input_mutated_ops_pass", [True]), \
    ("debug.aclgraph.disable_reinplace_inplaceable_ops_pass", [True])]
unsupport_aclgraphconfig_list = [("inference_config.dynamic_gears_merge_policy", ["product"]), \
    ("debug.fx_summary.type", ["csv"]), ("dump_config.enable_dump", [True]), \
    ("ge_config.export_compile_stat", ["1", "0"]), \
    ("export.experimental.auto_atc_config_generated", [True]), \
    ("export.experimental.enable_record_nn_module_stack", [True]), \
    ("ge_config.enable_single_stream", [True]), ("ge_config.oo_level", ["O1"]), \
    ("ge_config.oo_constant_folding", [True, False]), ("ge_config.oo_dead_code_elimination", [True, False]), \
    ("experimental_config.frozen_parameter", [True]), \
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
        for config_arg in unsupport_geconfig_list:
            if config_arg[0] + "._value" in config_dict.keys():
                warn_config = _get_warn_config(warn_config, config_arg, config_dict)
        warnings.warn("The following torchair config or properties may not take effect or report " + \
            "error in max-autotune mode: {warn_configs}".format(warn_configs=", ".join(warn_config)), \
                UserWarning)
    else:
        for config_arg in unsupport_aclgraphconfig_list:
            if config_arg[0] + "._value" in config_dict.keys():
                warn_config = _get_warn_config(warn_config, config_arg, config_dict)
        warnings.warn("The following torchair config or properties may not take effect or report " + \
            "error in reduce-overhead mode: {warn_configs}".format(warn_configs=", ".join(warn_config) + \
            ", cache_compile, set_dim_gears, dynamo_export, scope, npu_print"), UserWarning)


def _get_warn_config(warn_config, config_arg, config_dict):
    if config_arg[1] == []:
        warn_config.append("config." + config_arg[0] + ":" + str(config_dict[config_arg[0] + "._value"]))
    elif config_dict[config_arg[0] + "._value"] in config_arg[1]:
        warn_config.append("config." + config_arg[0] + ":" + str(config_dict[config_arg[0] + "._value"]))
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
