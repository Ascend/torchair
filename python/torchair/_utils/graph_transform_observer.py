import os
import threading
from typing import Callable
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torchair._ge_concrete_graph.utils import dump_graph
from torchair.ge._ge_graph import GeGraph
from torchair._utils.path_manager import PathManager
from torchair.core.utils import logger

debug_ctx = threading.local()


class GraphTransformObserver:
    '''
    Custom FX Pass observer to print debug information before and after FX Pass execution.
    '''
    __pass_count = 0

    def __init__(self, gm: GraphModule, example_inputs, config) -> None:
        self._gm = gm
        self.example_inputs = example_inputs
        self.config = config
        self._phase = getattr(debug_ctx, 'phase', 'default')
        self._path = os.path.join(DebugContext.current_path(), f"{self._phase}")
        self.dump_enabled = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1"
        self._logger = logger
       
    @classmethod
    def get_current_pass_count(cls):
        return cls.__pass_count

    @classmethod
    def reset_pass_count(cls):
        cls.__pass_count = 0
    
    def apply_gm_pass(self, pass_fn: Callable, pass_name: str):
        self._logger.debug('PASS_%s [%s] before [%s] execution, graph is %s',
                            GraphTransformObserver.__pass_count, self._phase, pass_name, self._gm.graph)
        res = pass_fn(self.gm, self.example_inputs, self.config)
        self._logger.debug('PASS_%s [%s] after [%s] execution, graph is %s', 
                           GraphTransformObserver.__pass_count, self._phase, pass_name, self._gm.graph)
        if not self.dump_enabled:
            return res
        self.dump_gm(self.gm, "graph_after_" + pass_name)
        return res


    def apply_gegraph_pass(self, pass_fn: Callable, ge: GeGraph, pass_name: str):
        """
        The Custom FX Pass input parameter is :
            apply_gegraph_pass(self, pass_fn, example_inputs, config):
        """
        res = pass_fn(ge)
        if not self.dump_enabled:
            return res
        self.dump_gegraph(ge, "graph_after_" + pass_name)
        return res
    
    def dump_gegraph(self, ge: GeGraph, file_name: str):
        """
        Non-pass function that requires sequential invocation with an ordinal number.
        """
        if not self.dump_enabled:
            return
        os.makedirs(self._path, exist_ok=True)
        file_type = self.config.debug.graph_dump.type.value or "pbtxt"
        file_name = (
        f"{GraphTransformObserver.__pass_count:03d}_aot_"
        f"{self._phase}_{file_name}.{file_type}"
        )   
        dump_graph(os.path.join(self._path, file_name), ge)
        GraphTransformObserver.__pass_count += 1
        return

    def dump_gm(self, gm: GraphModule, file_name: str):
        """
        Non-pass function that requires sequential invocation with an ordinal number.
        """
        if not self.dump_enabled:
            return
        dump_fx_safety(gm, os.path.join(
            self._path, f"{GraphTransformObserver.__pass_count:03d}_aot_{self._phase}_{file_name}.txt"))
        GraphTransformObserver.__pass_count += 1
        return


class DebugContext():
    """
    Debug context manager. Tracks debug phase and generates unique model debug paths.
    """
    COUNT = -1

    def __init__(self, phase: str) -> None:
        debug_ctx.phase = phase
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        GraphTransformObserver.reset_pass_count()
        if exc_type is None:
            return True
        else:
            raise exc_val.with_traceback(exc_tb)

    @classmethod
    def current_path(cls) -> str:
        return os.path.join(get_debug_dir(), "torchair", f"model__{cls.COUNT}")
    
    @classmethod
    def next_path(cls) -> str:
        cls.COUNT += 1
        return cls.current_path()
    

def wrap_compiler_phase(compiler: Callable, phase: str) -> Callable:
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
    inference_compiler = wrap_compiler_phase(inference_compiler, "inference")
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