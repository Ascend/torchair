"""Construct CompilerConfig configuration"""

__all__ = ["CompilerConfig"]

from typing import Any, Optional

from npugraph_ex.configs._option_base import OptionValue, CallableValue, NpuBaseConfig
from npugraph_ex.configs.aclgraph_config import _AclGraphConfig
from npugraph_ex.configs.debug_config import _DebugConfig
from npugraph_ex.configs.dump_config import _DataDumpConfig
from npugraph_ex.configs.experimental_config import _ExperimentalConfig
from npugraph_ex.configs.npugraphex_config import _NpuGraphExConfig


class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.debug = _DebugConfig()
        self.dump_config = _DataDumpConfig()
        self.experimental_config = _ExperimentalConfig()
        self.aclgraph_config = _AclGraphConfig()
        self.npugraphex_config = _NpuGraphExConfig()
        self.mode = OptionValue("npugraph_ex", ["npugraph_ex"])
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
        local_options, global_options = super().as_dict()
        if "post_grad_custom_pre_pass" in local_options.keys():
            local_options.pop("post_grad_custom_pre_pass")
        if "post_grad_custom_post_pass" in local_options.keys():
            local_options.pop("post_grad_custom_post_pass")
        return local_options, global_options


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
