from torchair.configs.option_base import OptionValue
from torchair.configs.option_base import NpuBaseConfig


class DataDumpConfig(NpuBaseConfig):
    """Config for data dump"""

    def __init__(self):
        self.enable_dump = OptionValue(False, [False, True])
        self.dump_path = OptionValue("./", None)
        self.dump_mode = OptionValue('all', ['input', 'output', 'all'])

        super(DataDumpConfig, self).__init__()

    def as_dict(self):
        dump_option = {}
        if self.enable_dump.value is True:
            dump_option['ge.exec.enableDump'] = '1'
            dump_option['ge.exec.dumpPath'] = self.dump_path.value
            dump_option['ge.exec.dumpMode'] = self.dump_mode.value
        return {}, dump_option
