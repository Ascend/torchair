import functools
from collections import defaultdict
from typing import List, Callable, Any, Dict, Tuple, Union

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._tensor import Tensor
from torch.utils._mode_utils import no_dispatch
from torch.fx import Interpreter
from torch.fx.node import Argument, Target
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.allowed_functions import is_builtin_callable

from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist
from torchair.core.utils import logger
from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph as ConcreteGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair.fx_summary import summarize_fx_graph
from torch._decomp import core_aten_decompositions, get_decompositions
aten = torch.ops.aten


def _unpack_meta_list(args):
    return [(arg.meta if (isinstance(arg, ValuePack)) else arg) for arg in args]


def _unpack_meta(args):
    unpacked = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and any(isinstance(v, ValuePack) for v in arg):
            arg = _unpack_meta_list(arg)
        if isinstance(arg, ValuePack):
            unpacked.append(arg.meta)
        else:
            unpacked.append(arg)
    return list(unpacked)


def _safe_str(x):
    try:
        if type(x) is torch.Tensor:
            return f"torch.Tensor(dtype={x.dtype}, size={list(x.size())}"
        return f"{x}"
    except Exception:
        return f"{type(x)}"


def trace_print(f):
    @functools.wraps(f)
    def inner(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        logger.debug(f'-------------------')
        logger.debug(f'target: {target}')
        for i, inp in enumerate(args):
            logger.debug(f'input {i}: {_safe_str(inp)}')
        for k, v in kwargs.items():
            logger.debug(f'input {k}: {_safe_str(v)}')
        result = f(self, target, args, kwargs)
        logger.debug(f'output {result}')
        return result

    return inner


class NpuGraphConverter(Interpreter):
    """
    Interpreter for collect npu graph meta from fx graph, such as sym of output, input shape ranges, etc.
    TODO: Add doc here
    """

    def __init__(self, *args, graph, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = graph

    def run(self, *args, **kwargs):
        with self._graph.context():
            super().run(*args, **kwargs)
            return self._graph

    def _unpack_npu(self, args):
        unpacked = []
        for arg in args:
            if isinstance(arg, (list, tuple)) and len(arg):
                if _is_symlist(arg):
                    arg = self._graph.parse_symlist(arg)
                else:
                    arg = [(v.npu if isinstance(v, ValuePack) else v)
                           for v in arg]

            if isinstance(arg, ValuePack):
                unpacked.append(arg.npu)
            else:
                unpacked.append(arg)
        return unpacked

    def _wrap(self, fn):
        def inner(target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
            func = getattr(super(NpuGraphConverter, self), fn)
            if is_builtin_callable(target):
                return func(target, args, kwargs)
            meta_outputs = func(target, _unpack_meta(args), kwargs)
            npu_outputs = self._graph.parse_node(
                target, self._unpack_npu(args), kwargs, meta_outputs)
            if isinstance(npu_outputs, (tuple, list)):
                return [ValuePack(k, v) for k, v in zip(meta_outputs, npu_outputs)]
            return ValuePack(meta_outputs, npu_outputs)

        return inner

    @trace_print
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_input = super().placeholder(target, args=args, kwargs=kwargs)
        npu_input = self._graph.parse_input(target, args, kwargs, meta_input)
        return ValuePack(meta_input, npu_input)

    @trace_print
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_function')(target, args, kwargs)

    @trace_print
    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_method')(target, args, kwargs)

    @trace_print
    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_module')(target, args, kwargs)

    @trace_print
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_output = super().placeholder(target, args=_unpack_meta(args), kwargs=kwargs)
        npu_output = self._graph.parse_output(
            target, args, kwargs, meta_output)
        return npu_output


def _summary(v):
    if isinstance(v, torch.Tensor):
        return f'{type(v)}({v.size()}, {v.dtype}, contiguous={v.is_contiguous()})'
    return f'{type(v)}({v})'


class _NpuFxCompiler:
    def __init__(self, compiler_config: CompilerConfig) -> None:
        self.config = compiler_config

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        logger.info(f'compiler inputs')
        for i, inp in enumerate(example_inputs):
            logger.info(f'  input {i}: {inp}')
        logger.info(f'  graph: {gm.graph}')

        summarize_fx_graph(gm, example_inputs, self.config.debug.fx_summary.full_path("summary"))

        concrete_graph: ConcreteGraphBase = NpuGraphConverter(
            gm, graph=ConcreteGraph(self.config)).run(*example_inputs)

        concrete_graph.dump(self.config.debug.graph_dump.full_path("dynamo"))

        logger.info(f'start compile graph: {concrete_graph}')
        concrete_graph.compile()
        logger.info(f'end compile graph: {concrete_graph}')

        def inference(*args, npu_compiled_gm, original_gm, **kwargs):
            logger.debug('runtime inputs')
            for i, inp in enumerate(args):
                logger.debug(f'  input {i}: {_summary(inp)}')
            for k, v in kwargs.items():
                logger.debug(f'  input {k}: {_summary(v)}')

            compiled_result = npu_compiled_gm(*args, **kwargs)

            logger.debug('runtime outputs')
            for i, inp in enumerate(compiled_result):
                logger.debug(f'  output {i}: {_summary(inp)}')

            return compiled_result

        return functools.partial(inference, npu_compiled_gm=concrete_graph, original_gm=gm)


def get_compiler(compiler_config: CompilerConfig = None):
    if compiler_config is None:
        compiler_config = CompilerConfig()
    return _NpuFxCompiler(compiler_config)


def _npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                 compiler_config: CompilerConfig = None, custom_decompositions: Dict = {}):
    decompositions = get_decompositions([])
    decompositions.update(custom_decompositions)
    compiler = get_compiler(compiler_config)
    return aot_module_simplified(gm, example_inputs, fw_compiler=compiler, decompositions=decompositions)


def get_npu_backend(*, compiler_config: CompilerConfig = None, custom_decompositions: Dict = {}):
    return functools.partial(_npu_backend, compiler_config=compiler_config, custom_decompositions=custom_decompositions)
