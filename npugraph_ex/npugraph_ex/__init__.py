__all__ = ['get_npu_backend', 'get_compiler', 'compile_fx', 'CompilerConfig', 'logger', 'register_replacement']

import atexit
import signal

from npugraph_ex.npu_fx_compiler import compile_fx, get_npu_backend, get_compiler
from npugraph_ex.configs.compiler_config import CompilerConfig
from npugraph_ex.core.utils import logger
from npugraph_ex.patterns.pattern_pass_manager import register_replacement
from npugraph_ex._utils.adjust_traceable_collective_remaps import adjust_traceable_collective_remaps
import npugraph_ex.inference
import npugraph_ex.ops
import npugraph_ex.scope

# before patch, backup function call for torch_npu.distributed.xxx
try:
    import torch_npu
    ALL_GATHER_INTO_TENSOR_UNEVEN = torch_npu.distributed.all_gather_into_tensor_uneven
    REDUCE_SCATTER_TENSOR_UNEVEN = torch_npu.distributed.reduce_scatter_tensor_uneven
except (ImportError, AttributeError) as e:
    ALL_GATHER_INTO_TENSOR_UNEVEN = None
    REDUCE_SCATTER_TENSOR_UNEVEN = None

adjust_traceable_collective_remaps()


def _initialize():
    from npugraph_ex._acl_concrete_graph import static_kernel
    static_kernel.cleanup_old_run_packages()


_initialize()


def _finalize():
    import torch
    from npugraph_ex._acl_concrete_graph import static_kernel

    torch._dynamo.reset()
    static_kernel.uninstall_static_kernel()


def _signal_handler(signum, frame):
    _finalize()
    signal.signal(signum, signal.SIG_IGN)
    import os
    os.kill(os.getpid(), signum)


atexit.register(_finalize)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
