from torch._inductor.pattern_matcher import (
    init_once_fakemode,
)
from torchair.core.utils import logger

registered_joint_graph_passes = {}


def register_joint_graph_pass(name):
    def registered_func(func):
        registered_joint_graph_passes[name] = func
        return func
    return registered_func


@init_once_fakemode
def lazy_init():
    import torchair._utils.npu_fx_passes.joint_graph_passes
    logger.info(f'Enable npu joint graph passes: {list(registered_joint_graph_passes.keys())}')
    for _, joint_graph_pass in registered_joint_graph_passes.items():
        joint_graph_pass()