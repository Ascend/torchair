"""Construct CompilerConfig configuration"""

from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import DeprecatedValue
from torchair.configs.option_base import NpuBaseConfig
from torchair.configs.aoe_config import AoeConfig
from torchair.configs.export_config import ExportConfig
from torchair.configs.debug_config import DebugConfig
from torchair.configs.dump_config import DataDumpConfig
from torchair.configs.fusion_config import FusionConfig
from torchair.configs.experimental_config import ExperimentalConfig


class CompilerConfig(NpuBaseConfig):
    """Set CompilerConfig configuration"""

    def __init__(self):
        self.debug = DebugConfig()

        self.aoe_config = AoeConfig()
        self.export = ExportConfig()
        self.dump_config = DataDumpConfig()
        self.fusion_config = FusionConfig()
        self.experimental_config = ExperimentalConfig()

        super(CompilerConfig, self).__init__()
