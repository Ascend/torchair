from torchair.configs._option_base import OptionValue, MustExistedPathValue
from torchair.configs._option_base import NpuBaseConfig

__all__ = []


class _DataDumpConfig(NpuBaseConfig):
    """Config for data dump"""

    def __init__(self):
        self.enable_dump = OptionValue(False, [False, True])
        self.dump_path = MustExistedPathValue("./")
        self.dump_mode = OptionValue('all', ['input', 'output', 'all'])
        self.quant_dumpable = OptionValue(False, [False, True])

        super(_DataDumpConfig, self).__init__()

    def as_dict(self):
        dump_option = {}
        if self.enable_dump:
            dump_option['ge.exec.enableDump'] = '1'
            dump_option['ge.exec.dumpPath'] = self.dump_path.value
            dump_option['ge.exec.dumpMode'] = self.dump_mode.value
            dump_option["ge.quant_dumpable"] = "1" if self.quant_dumpable else "0"
        return {}, dump_option
