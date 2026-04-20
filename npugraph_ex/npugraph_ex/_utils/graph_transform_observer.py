import os
import threading
from typing import Callable

from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule

from npugraph_ex._utils.path_manager import PathManager
from npugraph_ex.core.utils import logger

debug_ctx = threading.local()


class GraphTransformObserver:
    """
    Custom FX Pass observer to print debug information before and after FX Pass execution.
    """
    __file_count = 0

    def __init__(self, gm: GraphModule, example_inputs, config) -> None:
        self.gm = gm
        self.example_inputs = example_inputs
        self.config = config
        self._phase = getattr(debug_ctx, 'phase', 'default')
        self._path = os.path.join(DebugContext.current_path(), f"{self._phase}")
        self._dump_enabled = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1"
        self._logger = logger
       
    @classmethod
    def get_current_file_count(cls):
        return cls.__file_count

    @classmethod
    def reset_file_count(cls):
        cls.__file_count = 0

    def apply_gm_pass(self, pass_fn: Callable, pass_name: str, enable_log=False):
        if enable_log:
            self._logger.debug('[%s] before [%s] execution, graph is %s', self._phase, pass_name, self.gm.graph)

        res = pass_fn(self.gm, self.example_inputs, self.config)

        if enable_log:
            self._logger.debug('[%s] after [%s] execution, graph is %s', self._phase, pass_name, self.gm.graph)

        if not self._dump_enabled:
            return res
        self.dump_gm(self.gm, "graph_after_" + pass_name)
        return res


    def dump_gm(self, gm: GraphModule, file_name: str):
        """
        Non-pass function that requires sequential invocation with an ordinal number.
        """
        if not self._dump_enabled:
            return
        dump_fx_safety(gm, os.path.join(
            self._path, f"{GraphTransformObserver.__file_count:03d}_aot_{self._phase}_{file_name}.txt"))
        GraphTransformObserver.__file_count += 1
        return


class DebugContext():
    """
    Debug context manager. Tracks debug phase and generates unique model debug paths.
    """
    model_cnt = -1
    compile_fx_cnt = -1

    def __init__(self, phase: str) -> None:
        debug_ctx.phase = phase
        self._dump_enabled = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1"
        if self._dump_enabled and phase == "compile_fx":
            DebugContext.compile_fx_cnt += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphTransformObserver.reset_file_count()
        if self._dump_enabled:
            logger.debug(
                "[%s] Full debug info saved to: %s",
                getattr(debug_ctx, 'phase', 'default'),
                self.current_path()
            )
        if exc_type is None:
            return True
        else:
            raise exc_val.with_traceback(exc_tb)

    @classmethod
    def current_path(cls) -> str:
        phase = getattr(debug_ctx, 'phase', 'default')
        if phase == "compile_fx":
            return os.path.join(get_debug_dir(), "npugraph_ex", f"compile__{cls.compile_fx_cnt}")

        return os.path.join(get_debug_dir(), "npugraph_ex", f"model__{cls.model_cnt}")
    
    @classmethod
    def next_path(cls) -> str:
        cls.model_cnt += 1
        # Ensure phase is reset to None when entering the npu_backend
        debug_ctx.phase = None
        return cls.current_path()


def get_phase_path():
    return os.path.join(DebugContext.current_path(), getattr(debug_ctx, 'phase', 'default'))
    

def wrap_compiler_phase(compiler: Callable, phase: str) -> Callable:
    """
    Wrap a compiler function with a debug context for the specified phase.
    """
    if phase == "inference":
        phase = "forward"

    def wrapped(*args, **kwargs):
        with DebugContext(phase):
            return compiler(*args, **kwargs)
    return wrapped


def wrap_debug_compilers(
    fw_compiler: Callable,
    backward_compiler: Callable,
    inference_compiler: Callable,
    joint_compiler: Callable
):
    fw_compiler = wrap_compiler_phase(fw_compiler, "forward")
    backward_compiler = wrap_compiler_phase(backward_compiler, "backward")
    inference_compiler = wrap_compiler_phase(inference_compiler, "forward")
    joint_compiler = wrap_compiler_phase(joint_compiler, "joint")
    
    return fw_compiler, backward_compiler, inference_compiler, joint_compiler


def dump_fx_safety(gm, file_path: str = None):
    """
    Save the FX graph txt.
    """
    if file_path is None:
        logger.warning("dump file_path is None, skip dump graph.")
        return
    PathManager.check_path_writeable_and_safety(file_path)

    graph_str = gm.print_readable(False)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(graph_str)
