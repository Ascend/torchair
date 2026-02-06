__all__ = []

import os
from typing import Optional
import torch.distributed as dist
from torchair.configs._option_base import OptionValue, MustExistedPathValue, RegexValue
from torchair.configs._option_base import NpuBaseConfig, MustExistedFileAddr


class _DataDumpConfig(NpuBaseConfig):
    """Config for data dump"""

    def __init__(self):
        self.enable_dump = OptionValue(False, [False, True])
        self.dump_path = MustExistedPathValue("./")
        self.dump_mode = OptionValue('all', ['input', 'output', 'all'])
        self.quant_dumpable = OptionValue(False, [False, True])
        self.dump_step = RegexValue("", r'^(((\d+)|(\d+-{0,1}\d+))\|{0,1})*$', "0|1|2-5|6")
        self.dump_layer = RegexValue("", r'^[0-9a-zA-Z_" "/\\.]*$', "Mul_1 Add1 Conv2D_1")
        self.dump_data = OptionValue('tensor', ['tensor', 'stats'])
        self.dump_config_path = MustExistedFileAddr(None)
        self.data_dump_stage = OptionValue('optimized', ['original', 'optimized'])

        super(_DataDumpConfig, self).__init__()

    def full_path(self):
        path = self.dump_path.value
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            global_rank = dist.get_rank()
            path = os.path.join(self.dump_path.value, f'worldsize{dist.get_world_size()}_global_rank{global_rank}')
        else:
            path = os.path.join(self.dump_path.value, f'worldsize1_global_rank0')
        os.makedirs(path, exist_ok=True)
        return path

    def as_dict(self, mode: Optional[str] = "max-autotune"):
        dump_option = {}
        if self.dump_config_path.value is not None:
            dump_option['ge_dump_with_acl_config'] = self.dump_config_path.value
        if self.enable_dump and mode == "max-autotune":
            dump_option['ge.exec.enableDump'] = '1'
            dump_option['ge.exec.dumpPath'] = self.full_path()
            dump_option['ge.exec.dumpMode'] = self.dump_mode.value
            dump_option["ge.quant_dumpable"] = "1" if self.quant_dumpable else "0"
            if self.dump_step.value != "":
                dump_option['ge.exec.dumpStep'] = self.dump_step.value
            if self.dump_layer.value != "":
                dump_option['ge.exec.dumpLayer'] = self.dump_layer.value
            dump_option['ge.exec.dumpData'] = self.dump_data.value

        if self.enable_dump and mode in ("reduce-overhead", "npugraph_ex"):
            dump_option['aclgraph.enableDump'] = '1'
            dump_option['aclgraph.dumpPath'] = self.full_path()
            dump_option['aclgraph.dumpStage'] = self.data_dump_stage.value
        return {}, dump_option
