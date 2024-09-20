from torchair.configs._option_base import OptionValue, MustExistedPathValue, RegexValue
from torchair.configs._option_base import NpuBaseConfig

__all__ = []


class _DataDumpConfig(NpuBaseConfig):
    """Config for data dump"""

    def __init__(self):
        self.enable_dump = OptionValue(False, [False, True])
        self.dump_path = MustExistedPathValue("./")
        self.dump_mode = OptionValue('all', ['input', 'output', 'all'])
        self.quant_dumpable = OptionValue(False, [False, True])
        self.dump_step = RegexValue("", r'^(((\d+)|(\d+-{0,1}\d+))\|{0,1})*$', "0|1|2-5|6")
        self.dump_layer = RegexValue("", r'^[0-9a-zA-Z_" "/\\.]*$', "Mul_1 Add1 Conv2D_1")

        super(_DataDumpConfig, self).__init__()

    def as_dict(self):
        dump_option = {}
        if self.enable_dump:
            dump_option['ge.exec.enableDump'] = '1'
            dump_option['ge.exec.dumpPath'] = self.dump_path.value
            dump_option['ge.exec.dumpMode'] = self.dump_mode.value
            dump_option["ge.quant_dumpable"] = "1" if self.quant_dumpable else "0"
            if self.dump_step.value != "":
                dump_option['ge.exec.dumpStep'] = self.dump_step.value
            if self.dump_layer.value != "":
                dump_option['ge.exec.dumpLayer'] = self.dump_layer.value
        return {}, dump_option
