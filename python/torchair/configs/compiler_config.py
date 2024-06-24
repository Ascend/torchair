"""Construct CompilerConfig configuration"""

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


__all__ = ["CompilerConfig"]


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

        super(CompilerConfig, self).__init__()
