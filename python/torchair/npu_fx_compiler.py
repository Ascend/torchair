__all__ = ["get_npu_backend", "get_compiler"]

import functools
import operator
import math
import copy
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import logging
import sys
import os
import fcntl
import dataclasses
import types
from types import ModuleType
from packaging import version

import torch
from torch._subclasses.fake_tensor import is_fake
import torch.utils._pytree as pytree
from torch.fx import Interpreter
from torch.fx.node import Argument, Target
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import default_partition

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable

from torch.profiler import record_function
from torch.utils._mode_utils import no_dispatch
from torch._dynamo.utils import detect_fake_mode

from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist
from torchair.core.utils import logger
from torchair.ge._ge_graph import is_sym, _torch_tensor_to_ge_const
from torchair.configs.compiler_config import CompilerConfig, _check_config_support
from torchair.fx_summary import _summarize_fx_graph
from torchair.fx_dumper import _NpuFxDumper
from torchair._utils.custom_aot_functions import aot_module_simplified_joint
from torchair._utils import add_npu_patch, get_npu_default_decompositions
from torchair._utils.error_code import pretty_error_msg
from torchair._utils.graph_transform_observer import GraphTransformObserver, dump_fx_safety, \
    wrap_debug_compilers, DebugContext, get_phase_path
from torchair.inference._gear_utils import get_dim_gears, set_dim_gears, guard_gears_shape
from torchair.patterns.pattern_util import _apply_pattern_passes

aten = torch.ops.aten


def _unpack_meta_list(args):
    return [(arg.meta if (isinstance(arg, ValuePack)) else arg) for arg in args]


def _unpack_meta(args, kwargs):
    unpacked_args = []
    unpacked_kwargs = {}

    def _get_meta_part(arg):
        if isinstance(arg, (list, tuple)) and any(isinstance(v, ValuePack) for v in arg):
            return _unpack_meta_list(arg)
        elif isinstance(arg, ValuePack):
            return arg.meta
        else:
            return arg

    for arg in args:
        unpacked_args.append(_get_meta_part(arg))

    for key, value in kwargs.items():
        unpacked_kwargs[key] = _get_meta_part(value)

    return list(unpacked_args), unpacked_kwargs


def _safe_str(x):
    try:
        if isinstance(x, torch.Tensor):
            return f"torch.Tensor(dtype={x.dtype}, size={list(x.size())}"
        return f"{x}"
    except Exception:
        return f"{type(x)}"


def _is_binary_operator(target: Target):
    return target in (operator.add, operator.sub, operator.mul, operator.truediv, \
                      operator.floordiv, operator.pow, math.floor, operator.mod, math.ceil, operator.neg)


def _trace_print(f):
    @functools.wraps(f)
    def inner(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        logger.debug(f'-------------------')
        logger.debug('target: %s', target)
        for i, inp in enumerate(args):
            logger.debug('input %s: %s', i, _safe_str(inp))
        for k, v in kwargs.items():
            logger.debug('input %s: %s', k, _safe_str(v))
        result = f(self, target, args, kwargs)
        logger.debug('output %s', result)
        return result

    return inner


class _NpuGraphConverter(Interpreter):
    """
    Interpreter for collect npu graph meta from fx graph, such as sym of output, input shape ranges, etc.
    
    This class extends the Torch FX Interpreter to collect metadata necessary for constructing
    NPU computation graphs, such as symbolic shapes and input/output specifications.
    """

    def __init__(self, *args, graph, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = graph

    def run_node(self, n):
        """
        Overrides the default node execution to include graph context.

        Args:
            n (Node): The node to execute.

        Returns:
            Any: Result of node execution.
        """
        with self._graph.converter_context(node=n):
            return super().run_node(n)

    def run(self, *args, **kwargs):
        """
        Executes the interpreter and constructs the NPU graph.

        Args:
            *args: Positional inputs to the graph.
            **kwargs: Keyword inputs to the graph.

        Returns:
            GeConcreteGraph: Constructed NPU computation graph.
        """

        with self._graph.context():
            super().run(*args, **kwargs)
            return self._graph

    def _unpack_npu(self, args, kwargs):
        unpacked = []
        unpacked_kwargs = {}

        def _get_npu_part(arg):
            if isinstance(arg, (list, tuple)) and len(arg):
                if _is_symlist(arg):
                    arg = self._graph.parse_symlist(arg)
                else:
                    arg = [(v.npu if isinstance(v, ValuePack) else v)
                           for v in arg]
                return arg
            elif isinstance(arg, ValuePack):
                return arg.npu
            else:
                return arg

        for arg in args:
            unpacked.append(_get_npu_part(arg))

        for key, value in kwargs.items():
            unpacked_kwargs[key] = _get_npu_part(value)

        return unpacked, unpacked_kwargs

    def _wrap(self, fn):
        def inner(target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
            func = getattr(super(_NpuGraphConverter, self), fn)
            if is_builtin_callable(target) and not _is_binary_operator(target):
                return func(target, args, kwargs)
            args_meta, kwargs_meta = _unpack_meta(args, kwargs)
            fake_mode = detect_fake_mode(None)
            with fake_mode:
                meta_outputs = func(target, args_meta, kwargs_meta)
            args_npu, kwargs_npu = self._unpack_npu(args, kwargs)
            npu_outputs = self._graph.parse_node(target, args_npu, kwargs_npu, meta_outputs)
            return self._get_value_pack(meta_outputs, npu_outputs)

        return inner

    @_trace_print
    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handles placeholder nodes during interpretation.

        Args:
            target (Target): The target node.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Dict[str, Any]): Keyword arguments.

        Returns:
            ValuePack: Packed metadata and NPU value.
        """
        meta_input = super().placeholder(target, args=args, kwargs=kwargs)
        npu_input = self._graph.parse_input(target, args, kwargs, meta_input)
        return ValuePack(meta_input, npu_input)

    @_trace_print
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handles function call nodes during interpretation.

        Args:
            target (Target): The target function.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Dict[str, Any]): Keyword arguments.

        Returns:
            Any: Result of the function call.
        """
        return self._wrap('call_function')(target, args, kwargs)

    @_trace_print
    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handles method call nodes during interpretation.

        Args:
            target (Target): The target method.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Dict[str, Any]): Keyword arguments.

        Returns:
            Any: Result of the method call.
        """
        return self._wrap('call_method')(target, args, kwargs)

    @_trace_print
    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handles module call nodes during interpretation.

        Args:
            target (Target): The target module.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Dict[str, Any]): Keyword arguments.

        Returns:
            Any: Result of the module call.
        """
        return self._wrap('call_module')(target, args, kwargs)

    @_trace_print
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handles output nodes during interpretation.

        Args:
            target (Target): The output target.
            args (Tuple[Any, ...]): Positional arguments.
            kwargs (Dict[str, Any]): Keyword arguments.

        Returns:
            Any: Output value.
        """
        args_meta, kwargs_meta = _unpack_meta(args, kwargs)
        meta_output = super().placeholder(target, args=args_meta, kwargs=kwargs_meta)
        npu_output = self._graph.parse_output(
            target, args, kwargs, meta_output)
        return npu_output

    def _get_value_pack(self, meta_outputs, npu_outputs):
        if isinstance(npu_outputs, (tuple, list)):
            return [self._get_value_pack(k, v) for k, v in zip(meta_outputs, npu_outputs)]
        return ValuePack(meta_outputs, npu_outputs)


def _summary(v):
    if isinstance(v, torch.Tensor):
        return f'{type(v)}({v.size()}, {v.dtype}, contiguous={v.is_contiguous()})'
    return f'{type(v)}({v})'


def _view_to_reshape(graph_module: torch.fx.GraphModule, example_inputs=None, config=None):
    # Replace view ops in the GraphModule to reshape ops.
    for node in graph_module.graph.nodes:
        if node.target == torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape.default


def _optimize_fx(graph_module: torch.fx.GraphModule, config: CompilerConfig, observer: GraphTransformObserver):
    # More optimization passes here
    observer.gm = graph_module
    pre_func = config.post_grad_custom_pre_pass.value
    if pre_func is not None:
        observer.apply_gm_pass(pre_func, "post_grad_custom_pre_pass")

    if config.experimental_config.remove_noop_ops:
        observer.apply_gm_pass(_optimize_noop_ops, "optimize_noop_ops")

    from torchair.patterns._recover_view_inplace_pattern import recover_view_inplace_pattern
    observer.apply_gm_pass(recover_view_inplace_pattern, "recover_view_inplace_pattern")

    observer.apply_gm_pass(_optimize_sym_input, "optimize_sym_input")

    _add_stream_label_to_node_meta(graph_module)
    if config.experimental_config.pattern_fusion_pass:
        observer.apply_gm_pass(_apply_pattern_passes, "apply_pattern_passes")   

    observer.apply_gm_pass(_view_to_reshape, "view_to_reshape")

    post_func = config.post_grad_custom_post_pass.value
    if post_func is not None:
        observer.apply_gm_pass(post_func, "post_grad_custom_post_pass")
    logger.debug('after fx graph optimization, graph is %s', graph_module.graph)
    return graph_module


def _optimize_noop_ops(graph_module: torch.fx.GraphModule, example_inputs=None, config=None):
    try:
        from torch._inductor.fx_passes.post_grad import remove_noop_ops
        remove_noop_ops(graph_module.graph)
        logger.debug("After removing noop ops, graph is %s", graph_module.graph)
    except ImportError:
        logger.warning(("Skip removing noop ops; check if your PyTorch version is above 2.2.0 and ensure",
                        " the module torch._inductor.fx_passes.post_grad.remove_noop_ops exists"))


def _optimize_sym_input(graph_module: torch.fx.GraphModule, example_inputs=None, config=None):
    logger.debug('before sym input optimization, graph is %s', graph_module.graph)
    sym_input_list = []
    tensor_input_list = []
    data_idx = -1
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        data_idx = data_idx + 1
        if not hasattr(node, "meta"):
            # int placeholder does not have 'meta' attr or symbol, skip this case
            logger.debug('Find no meta attr placeholder node, placeholder index=%s, value=%s, type=%s',
                         data_idx, node, type(node).__name__)
            continue
        if 'val' not in node.meta:
            logger.debug('Find placeholder node with no val in meta, placeholder index=%s, value=%s, type=%s',
                         data_idx, node, type(node).__name__)
            continue
        if is_sym(node.meta['val']):
            sym_input_list.append(node)
        elif is_fake(node.meta['val']):
            tensor_input_list.append(node)
        else:
            # maybe int input, no need to process.
            pass

    for sym_node in sym_input_list:
        if sym_node.users == {}:
            continue
        for tensor_node in tensor_input_list:
            find_sym_in_tensor = False
            for i in range(len(tensor_node.meta['val'].size())):
                if str(sym_node.meta['val']) != str(tensor_node.meta['val'].size()[i]):
                    continue

                # find sym node is a dim of other fake tensor, replace it.
                with graph_module.graph.inserting_after(tensor_node):
                    sym_size_node = graph_module.graph.create_node(op="call_function", target=torch.ops.aten.sym_size,
                                                                   args=(tensor_node, i))
                    sym_node.replace_all_uses_with(sym_size_node, propagate_meta=True)
                    logger.debug('Replace node %s by inserting new node %s[op: %s'
                                 ', target: %s, meta: %s].', sym_node, sym_size_node, sym_size_node.op,
                                 sym_size_node.target, sym_size_node.meta)
                find_sym_in_tensor = True
                break
            if find_sym_in_tensor:
                break

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _add_stream_label_to_node_meta(graph_module: torch.fx.GraphModule):
    """
    Add stream labels to nodes in the FX graph based on their scope.
    - Nodes within a stream scope: meta["stream_label"] = corresponding stream label
    - Nodes outside a stream scope: meta["stream_label"] = None (default stream)
    """
    scope_enter_nodes_stack = []
    current_stream = None

    for node in graph_module.graph.nodes:
        if str(node.target) == "air.scope_enter.default":
            is_user_stream = len(node.args) > 0 and '_user_stream_label' in node.args[0]
            current_stream = node.args[1][0] if is_user_stream and len(node.args) > 1 else None
            node.meta["stream_label"] = current_stream
            scope_enter_nodes_stack.append(node)
        
        elif str(node.target) == "air.scope_exit.default":
            node.meta["stream_label"] = current_stream
            if scope_enter_nodes_stack:
                scope_enter_nodes_stack.pop()
                current_stream = scope_enter_nodes_stack[-1].args[1][0] if scope_enter_nodes_stack and len(scope_enter_nodes_stack[-1].args) > 1 else None

        else:
            node.meta["stream_label"] = current_stream

    graph_module.graph.lint()


def _valid_graph(graph_module):
    """
    After enabling users custom pass, it is necessary to perform checks here 
    to prevent users from introducing illegal graph modifications through these custom passes.
    """
    def _check_scope_enter_exit(graph_module):
        scope_enter_stack = []
        for node in graph_module.graph.nodes:
            if str(node.target) == "air.scope_enter.default":
                scope_enter_stack.append(node)
            elif str(node.target) == "air.scope_exit.default":
                if not scope_enter_stack:
                    raise RuntimeError(f"When you call the torch.ops.air.scope_exit operator: {node.name}, "
                                         f"you must first call the torch.ops.air.scope_enter operator, as they must be called in pairs. "
                                         f"Please check your code or your post_grad_custom_pre_pass post_grad_custom_post_pass!")
                scope_enter_stack.pop()

        if scope_enter_stack:
            args_list = [node.args for node in scope_enter_stack]
            raise RuntimeError(f"After you call the torch.ops.air.scope_enter operator, "
                                 f"there is no paired call to the torch.ops.air.scope_exit operator. "
                                 f"The parameters for these torch.ops.air.scope_enter calls are:{args_list}. "
                                 f"Please check your code or your post_grad_custom_pre_pass post_grad_custom_post_pass!")

    _check_scope_enter_exit(graph_module) 


@dataclasses.dataclass
class _CompiledFxArtifacts:
    """
    Artifacts for torchair compiled fx graph.
    """
    version: str = "0.1"
    py_code: str = None


class _CompiledFxGraph:
    """
    Wrapper for executing a graph module with runtime inputs.
    """

    def __init__(self, runner: Callable, config):
        self.config = config
        self.runner = runner
        self.run_kernel = None

    def __call__(self, *args, **kwargs):

        if self.run_kernel is None:
            if self.config.mode.value == "reduce-overhead":
                py_code = self.get_code()
                if not isinstance(py_code, str):
                    return py_code
                ge_mod = _compile_py_code(py_code)
                self.run_kernel = getattr(ge_mod, 'kernel')
            else:
                self.run_kernel = self.runner

        with record_function("npu_fx_compiler inference"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('runtime inputs')
                for i, inp in enumerate(args):
                    logger.debug('  input %s: %s', i, _summary(inp))
                for k, v in kwargs.items():
                    logger.debug('  input %s: %s', k, _summary(v))

            gm_result = self.run_kernel(*args, **kwargs)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('runtime outputs')
                for i, inp in enumerate(gm_result):
                    logger.debug('  output %s: %s', i, _summary(inp))

            return gm_result

    @classmethod
    def load_artifacts(cls, artifacts: _CompiledFxArtifacts):
        if artifacts.version != _CompiledFxArtifacts.version:
            raise RuntimeError(f'Unsupported artifacts version: {artifacts.version}, '
                               f'expected: {_CompiledFxArtifacts.version}.')
        compiled_mod = _compile_py_code(artifacts.py_code)
        return _CompiledFxGraph(getattr(compiled_mod, 'kernel'))

    def dump_artifacts(self):
        if not hasattr(self.runner, 'codegen'):
            raise RuntimeError(f'Compiled fx type {self.runner} does not support serialize.')
        code = self.runner.codegen(extend_config={})
        return _CompiledFxArtifacts(py_code=code)

    @pretty_error_msg
    def get_code(self, extend_config=None):
        if not hasattr(self.runner, 'codegen'):
            logger.warning(f'When enable FX Graph summarizing or dumping, codegen is unsupported.')
            return self.runner

        py_code = self.runner.codegen(extend_config=extend_config, enable_cache=True)
        if py_code is None:
            logger.warning(f'There are some configurations that cannot be supported by codegen, skipping codegen.')
            return self.runner
        logger.debug('Codegen for %s successfully, code:\n%s.', self.runner.graph.name, py_code)
        if self.config.mode.value == "reduce-overhead":
            _dump_run_codegen(py_code)
        return py_code


_GLOBAL_GRAPH_ID = 0


def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID


class _NpuFxCompiler:
    """
    Main compiler class for converting FX graphs to NPU-compatible graphs.
    """

    def __init__(self, compiler_config: CompilerConfig) -> None:
        self.config = compiler_config

    @pretty_error_msg
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        Compiles the FX graph into an NPU-compatible graph.

        Args:
            gm (torch.fx.GraphModule): The FX graph module to compile.
            example_inputs (List[torch.Tensor]): Example inputs for compilation.

        Returns:
            _CompiledFxGraph: Runner wrapping the compiled graph.
        """

        return self._get_compiled_gm(gm, example_inputs)


    def _get_compiled_gm(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        Internal method to generate the compiled graph module.

        Args:
            gm (torch.fx.GraphModule): The FX graph module.
            example_inputs (List[torch.Tensor]): Example inputs.

        Returns:
            _CompiledFxGraph: Runner wrapping the compiled graph.
        """
        if int(self.config.export.experimental.enable_lite_export.value):
            from torchair._ge_concrete_graph.ge_converter import lite

        if self.config.debug.fx_summary.enabled and self.config.mode.value == "reduce-overhead":
            logger.warning(f"The fx_summary csv files will not be generated in reduce-overhead mode.")

        observer = GraphTransformObserver(gm, example_inputs, self.config)
        observer.dump_gm(gm, "graph")

        if self.config.debug.fx_summary.enabled and self.config.mode.value == "max-autotune":
            _summarize_fx_graph(
                gm, example_inputs, self.config.debug.fx_summary.full_path("summary"))
            if self.config.debug.fx_summary.skip_compile:
                logger.warning(f'When summarizing FX Graph, npu compilation will be skipped, '
                               'and FALLBACK to EAGER execution to ensure the integrity of the analysis data. '
                               'Once the analysis is complete, please make sure to disable the summary config '
                               'to ensure that the graph is compiled and executed.')
                return _CompiledFxGraph(gm, self.config)

        if self.config.debug.data_dump.enabled:
            logger.warning(f'When dumping data of FX Graph, npu run will be skipped, '
                           'and FALLBACK to EAGER execution, once dump finished, please make sure to disable '
                           'the data dump config to ensure that the graph is compiled and executed.')
            data_dumper = _NpuFxDumper(gm, config=self.config.debug.data_dump,
                                       name="graph_" + str(_next_unique_graph_id()))
            return _CompiledFxGraph(data_dumper, self.config)

        return _CompiledFxGraph(self._gen_compiled_gm(gm, example_inputs, observer), self.config)

    def _gen_compiled_gm(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], observer: GraphTransformObserver):
        logger.info(f'compiler inputs')
        for i, inp in enumerate(example_inputs):
            logger.info('  input %s: %s', i, inp)
        logger.info('  graph: %s', gm.graph)

        # to temporarily fix weight_quant_batchmatmul bug
        if "torch_npu" in sys.modules:
            for n in gm.graph.nodes:
                if n.op == "call_function" and str(n.target) == "npu.npu_weight_quant_batchmatmul.default":
                    self.config.experimental_config.enable_view_optimize = False
                    logger.warning(f'To temporarily fix weight_quant_batchmatmul bug, close enable_view_optimize.')
                    break

        # generate different concrete graph based on config
        with no_dispatch():
            mutable_gm = copy.deepcopy(gm)
        if self.config.mode.value == "max-autotune":
            from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
            graph = GeConcreteGraph(self.config, name="graph_" + str(_next_unique_graph_id()))
        elif self.config.mode.value == "reduce-overhead":
            from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
            graph = AclConcreteGraph(self.config,
                                     name="graph_" + str(_next_unique_graph_id()),
                                     pool=self.config.aclgraph_config.use_custom_pool)
        else:
            raise ValueError(f"Unsupported npu backend mode: {self.config.mode.value}.")

        # do common optimization for fx graph based on config
        optimized_gm = _optimize_fx(mutable_gm, self.config, observer)
        _valid_graph(optimized_gm)
        graph.save_fx_graph(optimized_gm)

        concrete_graph: ConcreteGraphBase = _NpuGraphConverter(
            optimized_gm, graph=graph, garbage_collect_values=False).run(*example_inputs)

        if self.config.debug.graph_dump.enabled and not self.config.export.export_mode:
            concrete_graph.dump(self.config.debug.graph_dump.full_path(f"dynamo_original_graph_{_GLOBAL_GRAPH_ID}"))
        
        # optimize different concrete graph for ge or acl.
        concrete_graph.optimize_graph_without_runtime(*example_inputs, observer=observer)

        if self.config.debug.run_eagerly:
            logger.warning(f'When using debug.run_eagerly=True, npu compiler will be skipped, '
                           'and FALLBACK to EAGER execution, once running finished, please make sure to disable '
                           'the debug.run_eagerly=True config to ensure that the graph is compiled and executed.')
            if not graph.fx_graph:
                raise RuntimeError('When using debug.run_eagerly=True, the FX graph should not be None.')
            return graph.fx_graph

        return concrete_graph


def get_compiler(compiler_config: CompilerConfig = None):
    """
    Retrieves the NPU compiler instance.

    Args:
        compiler_config (CompilerConfig, optional): Compiler configuration. Defaults to None.

    Returns:
        _NpuFxCompiler: NPU compiler instance.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    return _NpuFxCompiler(compiler_config)


def _npu_joint_graph_passes(graph):
    from torch._inductor.fx_passes.joint_graph import joint_graph_passes
    joint_graph_passes(graph)


def _get_partition_fn(compiler_config: CompilerConfig):
    def partition_fn(graph: torch.fx.GraphModule, joint_inputs, **kwargs):
        _npu_joint_graph_passes(graph)
        return default_partition(graph, joint_inputs, **kwargs)

    if compiler_config.experimental_config.npu_fx_pass:
        return partition_fn
    return default_partition


def _wrap_compiler(compiler: Callable, compiler_config: CompilerConfig):
    @functools.wraps(compiler)
    def wrapped_compiler(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
            is_inference: bool
    ):
        if is_inference and compiler_config.experimental_config.npu_fx_pass:
            _npu_joint_graph_passes(gm)
        return compiler(gm, example_inputs)

    @functools.wraps(compiler)
    def joint_compiler(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor]
    ):
        if compiler_config.experimental_config.npu_fx_pass:
            _npu_joint_graph_passes(gm)
        return compiler(gm, example_inputs)

    fw_compiler = functools.partial(wrapped_compiler, is_inference=False)
    inference_compiler = functools.partial(wrapped_compiler, is_inference=True)
    return fw_compiler, inference_compiler, joint_compiler


def _get_inputs_custom_attr(example_inputs: List[torch.Tensor]):
    inputs_attr = {}
    for i, t in enumerate(example_inputs):
        attr = {}
        dim_gears = get_dim_gears(t)
        if dim_gears is not None:
            attr["dim_gears"] = dim_gears
        if hasattr(t, "_dynamo_static_input_type"):
            attr["_dynamo_static_input_type"] = t._dynamo_static_input_type
        if isinstance(t, torch.nn.Parameter):
            attr["_torchair_is_parameter"] = True
        if attr:
            inputs_attr[i - len(example_inputs)] = attr
    return inputs_attr


def _set_inputs_custom_attr(example_inputs: List[torch.Tensor], inputs_custom_attr: Dict[int, Dict]):
    for i, attr in inputs_custom_attr.items():
        for k, v in attr.items():
            if k == "dim_gears":
                set_dim_gears(example_inputs[i], v)
            else:
                setattr(example_inputs[i], k, v)
    guard_gears_shape(example_inputs)


def _set_inputs_custom_attr_to_compiler(compiler: Callable, inputs_custom_attr: Dict[int, Dict]):
    @functools.wraps(compiler)
    def _warp_custom_attr_compiler(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
    ):
        _set_inputs_custom_attr(example_inputs, inputs_custom_attr)
        return compiler(gm, example_inputs)

    return _warp_custom_attr_compiler


def _npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                 compiler_config: CompilerConfig = None, decompositions: Dict = {}):
    """
    Backend entry point for NPU compilation.

    Args:
        gm (torch.fx.GraphModule): The graph module to compile.
        example_inputs (List[torch.Tensor]): Example inputs.
        compiler_config (CompilerConfig, optional): Compiler configuration. Defaults to None.
        decompositions (Dict, optional): Custom decomposition rules. Defaults to {}.

    Returns:
        Any: Compiled graph runner.
    """
    
    if compiler_config is None:
        compiler_config = CompilerConfig()
    compiler = get_compiler(compiler_config)

    if os.getenv("TORCH_COMPILE_DEBUG", "0") == "1":
        folder_path = DebugContext.next_path()
        dump_fx_safety(gm, os.path.join(folder_path, "dynamo_out_graph.txt"))

    fw_compiler, inference_compiler, joint_compiler = _wrap_compiler(compiler, compiler_config)

    inputs_custom_attr = _get_inputs_custom_attr(example_inputs)
    fw_compiler = _set_inputs_custom_attr_to_compiler(fw_compiler, inputs_custom_attr)
    inference_compiler = _set_inputs_custom_attr_to_compiler(inference_compiler, inputs_custom_attr)

    partition_fn = _get_partition_fn(compiler_config)

    fw_compiler, compiler, inference_compiler, joint_compiler = wrap_debug_compilers(
        fw_compiler, compiler, inference_compiler, joint_compiler)

    if compiler_config.experimental_config.aot_config_enable_joint_graph:
        output_loss_index = int(compiler_config.experimental_config.aot_config_output_loss_index.value)
        return aot_module_simplified_joint(gm, example_inputs,
                                           compiler=joint_compiler, decompositions=decompositions,
                                           output_loss_index=output_loss_index)

    keep_inference_input_mutations = bool(compiler_config.experimental_config.keep_inference_input_mutations)
    return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler, bw_compiler=compiler,
                                 decompositions=decompositions, partition_fn=partition_fn,
                                 keep_inference_input_mutations=keep_inference_input_mutations,
                                 inference_compiler=inference_compiler)


def get_npu_backend(*, compiler_config: CompilerConfig = None, custom_decompositions: Dict = {}):
    """
    Retrieves the NPU backend compiler with custom configurations.

    Args:
        compiler_config (CompilerConfig, optional): Compiler configuration. Defaults to None.
        custom_decompositions (Dict, optional): Custom decomposition rules. Defaults to {}.

    Returns:
        Callable: Backend compiler function.
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()
    _check_config_support(compiler_config)
    decompositions = get_npu_default_decompositions()
    decompositions.update(custom_decompositions)

    add_npu_patch(decompositions, compiler_config)
    return functools.partial(_npu_backend, compiler_config=compiler_config, decompositions=decompositions)


def _dump_run_codegen(py_code: str):
    if os.getenv("TORCH_COMPILE_DEBUG", "0") == "1":

        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()
        input_code.writelines(["", "", 'if __name__ == "__main__":'])
        with input_code.indent():
            input_code.writeline("main()")
            py_codegen_dump = py_code + input_code.getvalue()

        # 进行生成dump文件，保存codegen内容
        file_name = os.path.join(get_phase_path(), "output_code.py")
        output_code_path = os.path.realpath(file_name)
        os.makedirs(os.path.dirname(output_code_path), exist_ok=True)
        with open(output_code_path, "w") as f:
            from torchair.inference._cache_compiler import file_lock
            with file_lock(f, fcntl.LOCK_EX):
                f.write(py_codegen_dump)
                os.chmod(f.fileno(), 0o600)


def _compile_py_code(py_code: str):
    ge_mod = ModuleType('ge_mod')
    exec(compile(py_code, '<string>', 'exec'), ge_mod.__dict__, ge_mod.__dict__)
    return ge_mod
