"""Construct CompilerConfig configuration"""

__all__ = ["CompilerConfig"]

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
