from torchair.configs._option_base import OptionValue
from torchair.configs._option_base import NpuBaseConfig

__all__ = []


class _GEConfig(NpuBaseConfig):
    """Config for ge option"""

    def __init__(self):
        self.enable_single_stream = OptionValue(False, [True, False])
        self.oo_level = OptionValue("O3", ["O1", "O3"])
        self.oo_constant_folding = OptionValue(None, [True, False])
        self.oo_dead_code_elimination = OptionValue(None, [True, False])

        super(_GEConfig, self).__init__()

    def as_dict(self):
        global_option = {}
        local_option = {}
        sorting_strategy_dict = {"BFS": "0", "DFS": "1", "RDFS": "2", "StableRDFS": "3"}

        global_option["ge.enableSingleStream"] = "true" if self.enable_single_stream else "false"
        global_option["ge.oo.level"] = self.oo_level.value
        if self.oo_constant_folding.value is not None:
            global_option["ge.oo.constant_folding"] = "true" if self.oo_constant_folding else "false"
        if self.oo_dead_code_elimination.value is not None:
            global_option["ge.oo.dead_code_elimination"] = "true" \
                if self.oo_dead_code_elimination else "false"
        if self.oo_level.value == "O1":
            local_option["ge.topoSortingMode"] = sorting_strategy_dict["StableRDFS"]
        return local_option, global_option