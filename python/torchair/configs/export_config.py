from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class ExportConfig(NpuBaseConfig):
    """Config for export"""

    def __init__(self):
        self.export_mode = False
        self.export_path_dir = None
        self.export_name = None
        self.weight_name = None
        self.inputs_name = None

    def as_dict(self):
        if self.export_mode:
            export_option = {}
            export_option['export_path_dir'] = self.export_path_dir
            export_option['export_name'] = self.export_name
            return export_option
        return {}

