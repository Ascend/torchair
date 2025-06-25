__all__ = ["get_npu_backend", "get_compiler"]

import functools
import operator
import math
import copy
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import logging
import sys

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
from torchair.inference._gear_utils import get_dim_gears, set_dim_gears, guard_gears_shape


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
                      operator.floordiv, operator.pow, math.floor, operator.mod, math.ceil)


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
    TODO: Add doc here
    """

    def __init__(self, *args, graph, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = graph

    def run_node(self, n):
        with self._graph.converter_context(node=n):
            return super().run_node(n)

    def run(self, *args, **kwargs):
        optimized_fx = _optimize_fx(self.module)
        self._graph.save_fx_graph(optimized_fx)

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
        meta_input = super().placeholder(target, args=args, kwargs=kwargs)
        npu_input = self._graph.parse_input(target, args, kwargs, meta_input)
        return ValuePack(meta_input, npu_input)

    @_trace_print
    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_function')(target, args, kwargs)

    @_trace_print
    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_method')(target, args, kwargs)

    @_trace_print
    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_module')(target, args, kwargs)

    @_trace_print
    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
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


def _optimize_fx(graph_module: torch.fx.GraphModule):
    # More optimization passes here
    graph_module = _optimize_sym_input(graph_module)
    logger.debug('after sym input optimization, graph is %s', graph_module.graph)
    return graph_module


def _optimize_sym_input(graph_module: torch.fx.GraphModule):
    logger.debug('before sym input optimization, graph is %s', graph_module.graph)
    sym_input_list = []
    tensor_input_list = []
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
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


class _GmRunner:
    def __init__(self, runner: Callable):
        self.runner = runner

    def __call__(self, *args, **kwargs):
        with record_function("npu_fx_compiler inference"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('runtime inputs')
                for i, inp in enumerate(args):
                    logger.debug('  input %s: %s', i, _summary(inp))
                for k, v in kwargs.items():
                    logger.debug('  input %s: %s', k, _summary(v))

            gm_result = self.runner(*args, **kwargs)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('runtime outputs')
                for i, inp in enumerate(gm_result):
                    logger.debug('  output %s: %s', i, _summary(inp))

            return gm_result


_GLOBAL_GRAPH_ID = 0


def _next_unique_graph_id():
    global _GLOBAL_GRAPH_ID
    _GLOBAL_GRAPH_ID += 1
    return _GLOBAL_GRAPH_ID


class _NpuFxCompiler:
    def __init__(self, compiler_config: CompilerConfig) -> None:
        self.config = compiler_config

    @pretty_error_msg
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return self._get_compiled_gm(gm, example_inputs)

    @pretty_error_msg
    def codegen(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], *,
                extend_config: Optional[dict] = None):
        gm_runner = self._get_compiled_gm(gm, example_inputs)
        if not hasattr(gm_runner.runner, 'codegen'):
            logger.warning(f'When enable FX Graph summarizing or dumping, codegen is unsupported.')
            return gm_runner

        code = gm_runner.runner.codegen(extend_config, enable_cache=True)
        if code is None:
            logger.warning(f'There are some configurations that cannot be supported by codegen, skipping codegen.')
            return gm_runner

        logger.debug('Codegen for %s successfully, code:\n%s.', gm_runner.runner.graph.name, code)
        return code

    def _get_compiled_gm(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        if int(self.config.export.experimental.enable_lite_export.value):
            from torchair._ge_concrete_graph.ge_converter import lite
        
        if self.config.debug.fx_summary.enabled and self.config.mode.value == "reduce-overhead":
            logger.warning(f"The fx_summary csv files will not be generated in reduce-overhead mode.")
        
        if self.config.debug.fx_summary.enabled and self.config.mode.value == "max-autotune":
            _summarize_fx_graph(
                gm, example_inputs, self.config.debug.fx_summary.full_path("summary"))
            if self.config.debug.fx_summary.skip_compile:
                logger.warning(f'When summarizing FX Graph, npu compilation will be skipped, '
                               'and FALLBACK to EAGER execution to ensure the integrity of the analysis data. '
                               'Once the analysis is complete, please make sure to disable the summary config '
                               'to ensure that the graph is compiled and executed.')
                return _GmRunner(gm)

        if self.config.debug.data_dump.enabled:
            logger.warning(f'When dumping data of FX Graph, npu run will be skipped, '
                           'and FALLBACK to EAGER execution, once dump finished, please make sure to disable '
                           'the data dump config to ensure that the graph is compiled and executed.')
            data_dumper = _NpuFxDumper(gm, config=self.config.debug.data_dump)
            return _GmRunner(data_dumper)

        return _GmRunner(self._gen_compiled_gm(gm, example_inputs))

    def _gen_compiled_gm(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
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

        with no_dispatch():
            mutable_gm = copy.deepcopy(gm)
        if self.config.mode.value == "max-autotune":
            from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph
            graph = GeConcreteGraph(self.config, name="graph_" + str(_next_unique_graph_id()))
        elif self.config.mode.value == "reduce-overhead":
            from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
            graph = AclConcreteGraph(self.config, name="graph_" + str(_next_unique_graph_id()))
        else:
            raise ValueError(f"Unsupported npu backend mode: {self.config.mode.value}.")
        concrete_graph: ConcreteGraphBase = _NpuGraphConverter(
            mutable_gm, graph=graph, garbage_collect_values=False).run(*example_inputs)

        if self.config.debug.graph_dump.enabled and not self.config.export.export_mode:
            concrete_graph.dump(self.config.debug.graph_dump.full_path(f"dynamo_original_graph_{_GLOBAL_GRAPH_ID}"))

        concrete_graph.optimize_graph_without_runtime(*example_inputs)

        if self.config.debug.run_eagerly:
            logger.warning(f'When using debug.run_eagerly=True, npu compiler will be skipped, '
                           'and FALLBACK to EAGER execution, once running finished, please make sure to disable '
                           'the debug.run_eagerly=True config to ensure that the graph is compiled and executed.')
            if not graph.fx_graph:
                raise RuntimeError('When using debug.run_eagerly=True, the FX graph should not be None.')
            return graph.fx_graph

        return concrete_graph


def get_compiler(compiler_config: CompilerConfig = None):
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


def _set_gear_to_compiler(compiler: Callable, compiler_config: CompilerConfig, input_dim_gears: Dict[int, List[int]]):
    @functools.wraps(compiler)
    def gear_compiler(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
    ):
        for i, dim_gears in input_dim_gears.items():
            set_dim_gears(example_inputs[i], dim_gears)
        guard_gears_shape(example_inputs)
        return compiler(gm, example_inputs)
    return gear_compiler


def _npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                 compiler_config: CompilerConfig = None, decompositions: Dict = {}):
    if compiler_config is None:
        compiler_config = CompilerConfig()
    compiler = get_compiler(compiler_config)

    input_dim_gears = dict()
    for i, t in enumerate(example_inputs):
        dim_gears = get_dim_gears(t)
        if dim_gears is not None:
            input_dim_gears[i - len(example_inputs)] = dim_gears

    fw_compiler, inference_compiler, joint_compiler = _wrap_compiler(compiler, compiler_config)
    fw_compiler = _set_gear_to_compiler(fw_compiler, compiler_config, input_dim_gears)
    inference_compiler = _set_gear_to_compiler(inference_compiler, compiler_config, input_dim_gears)

    partition_fn = _get_partition_fn(compiler_config)
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
    if compiler_config is None:
        compiler_config = CompilerConfig()

    _check_config_support(compiler_config)
    decompositions = get_npu_default_decompositions()
    decompositions.update(custom_decompositions)

    add_npu_patch(decompositions, compiler_config)

    return functools.partial(_npu_backend, compiler_config=compiler_config, decompositions=decompositions)