__all__ = ['get_npu_backend', 'get_compiler', 'compile_fx', 'CompilerConfig', 'logger', 'register_replacement']

import atexit

from .npu_fx_compiler import compile_fx, get_npu_backend, get_compiler
from .configs.compiler_config import CompilerConfig
from .core.utils import logger
from .patterns.pattern_pass_manager import register_replacement
from ._utils.adjust_traceable_collective_remaps import adjust_traceable_collective_remaps
from . import inference
from . import ops
from . import scope

# before patch, backup function call for torch_npu.distributed.xxx
try:
    import torch_npu
    ALL_GATHER_INTO_TENSOR_UNEVEN = torch_npu.distributed.all_gather_into_tensor_uneven
    REDUCE_SCATTER_TENSOR_UNEVEN = torch_npu.distributed.reduce_scatter_tensor_uneven
except (ImportError, AttributeError) as e:
    ALL_GATHER_INTO_TENSOR_UNEVEN = None
    REDUCE_SCATTER_TENSOR_UNEVEN = None

adjust_traceable_collective_remaps()


def _finalize_graph_engine():
    import torch
    from npugraph_ex._acl_concrete_graph import static_kernel

    torch._dynamo.reset()
    static_kernel.uninstall_static_kernel()


atexit.register(_finalize_graph_engine)