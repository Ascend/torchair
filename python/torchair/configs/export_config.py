__all__ = []

from torchair.configs._option_base import OptionValue, MustExistedPathValue
from torchair.configs._option_base import NpuBaseConfig


class _ExperimentalConfig(NpuBaseConfig):
    """Config for experimental"""

    def __init__(self):
        self.enable_record_nn_module_stack = OptionValue(False, [True, False])
        self.auto_atc_config_generated = OptionValue(False, [True, False])
        self.enable_lite_export = OptionValue(False, [True, False])
        super(_ExperimentalConfig, self).__init__()


class _ExportConfig(NpuBaseConfig):
    """Config for export"""

    def __init__(self):
        self.export_mode = OptionValue(False, [False, True])
        self.export_path_dir = MustExistedPathValue("./")
        self.export_name = None
        self.weight_name = None
        self.inputs_name = None
        self.experimental = _ExperimentalConfig()

    def as_dict(self):
        if self.export_mode:
            export_option = {'export_path_dir': self.export_path_dir, 'export_name': self.export_name}
            return export_option, {}
        return {}, {}
