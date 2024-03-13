import functools
import operator
import copy
from typing import List, Callable, Any, Dict, Tuple, Union
import logging
import sys

import torch
from torch._C import DispatchKey
from torch._subclasses.fake_tensor import is_fake
import torch.utils._pytree as pytree
from torch.fx import Interpreter
from torch.fx.node import Argument, Target
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import default_partition
from torch._inductor.fx_passes.joint_graph import joint_graph_passes

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable
    
from torch._decomp import get_decompositions, decomposition_table
from torch.profiler import record_function
from torch.utils._mode_utils import no_dispatch
from torch._dynamo.utils import detect_fake_mode

from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist
from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_graph import is_sym, _torch_tensor_to_ge_const
from torchair.ge_concrete_graph.utils import get_used_syms_in_meta
from torchair.ge_concrete_graph.fx2ge_converter import GeConcreteGraph as ConcreteGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair.configs.aot_config import AotConfig
from torchair.fx_summary import summarize_fx_graph
from torchair.fx_dumper import NpuFxDumper
from torchair.utils.custom_aot_functions import aot_module_simplified_joint
from torchair.utils import add_npu_patch
aten = torch.ops.aten


def get_default_decompositions():
    default_decompositions = []
    if "torchair.ge_concrete_graph.ge_converter.experimental.hcom_alltoall" in sys.modules:
        default_decompositions.append(torch.ops.npu_define.all_to_all_single)
        default_decompositions.append(torch.ops.npu_define.all_to_all)
    if "torchair.ge_concrete_graph.ge_converter.experimental.hcom_allgather" in sys.modules:
        default_decompositions.append(torch.ops.npu_define.allgather)
    return default_decompositions


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
        if type(x) is torch.Tensor:
            return f"torch.Tensor(dtype={x.dtype}, size={list(x.size())}"
        return f"{x}"
    except Exception:
        return f"{type(x)}"


def _is_binary_operator(target: Target):
    return target in (operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv, operator.pow)


def make_real_tensor_like(meta_outputs):
    if isinstance(meta_outputs, (tuple, list)):
        return [make_real_tensor_like(v) for v in meta_outputs]
    with no_dispatch():
        empty_tensor = torch.empty(meta_outputs.size(), dtype=meta_outputs.dtype)
        ge_empty = _torch_tensor_to_ge_const(empty_tensor)
        ge_empty.set_meta(meta_outputs)
        return ge_empty


def is_zero_element_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and 0 in tensor.size() and not get_used_syms_in_meta(tensor)


def flatten_meta_outputs(meta_outputs):
    flat_outputs = []
    if not isinstance(meta_outputs, (tuple, list)):
        meta_outputs = [meta_outputs]
    for i in meta_outputs:
        if isinstance(i, (tuple, list)):
            flat_outputs.extend(flatten_meta_outputs(i))
        else:
            flat_outputs.append(i)
    return flat_outputs


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

    def run_node(self, n):
        if n.stack_trace is not None:
            file_line = n.stack_trace.split(' File ')[-1].replace('\n', '')
            if file_line not in self._graph.graph._python_code:
                self._graph.graph._python_code += f'\n# File {file_line}\n'
            self._graph.graph._python_code += \
                f'## FX Code: ' \
                f'{self._graph.graph.format_python_code(n.name, n._pretty_print_target(n.target), None, n.args, n.kwargs)}\n'

        with self._graph.converter_context(node=n):
            return super().run_node(n)

    def run(self, *args, **kwargs):
        flat_args, _ = pytree.tree_flatten((args, kwargs))
        optimize_fx_input = _optimize_fx(flat_args, self.module, self._graph)

        with self._graph.context():
            super().run(*optimize_fx_input)
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
            func = getattr(super(NpuGraphConverter, self), fn)
            if is_builtin_callable(target) and not _is_binary_operator(target):
                return func(target, args, kwargs)
            args_meta, kwargs_meta = _unpack_meta(args, kwargs)
            fake_mode = detect_fake_mode(None)
            with fake_mode:
                meta_outputs = func(target, args_meta, kwargs_meta)
            args_npu, kwargs_npu = self._unpack_npu(args, kwargs)
            if all([is_zero_element_tensor(t) for t in flatten_meta_outputs(meta_outputs)]):
                npu_outputs = make_real_tensor_like(meta_outputs)
            else:
                npu_outputs = self._graph.parse_node(target, args_npu, kwargs_npu, meta_outputs)
            return self._get_value_pack(meta_outputs, npu_outputs)

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


def _get_placeholder_nodes(gm: torch.fx.GraphModule):
    placeholder_nodes_list = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes_list.append(node)
    return placeholder_nodes_list


def _remove_sym_from_shape(gm: torch.fx.GraphModule):
    '''
    函数功能： 消除图中input节点symint部分, 此symint需要能通过input tensor中的某一维shape表达
    目的： 在线场景为了性能， 离线场景功能要求必须消除dynamo框架产生的输入
    input中的sx有如下来源
    1、sx为shape的某一维度被泛化--->该输入可以被优化, 处理为对该输入做sym_size的动作, 并全图将原来的sx替换
    2、由其他sym组成的syms,如 2*s0 - s1 也是一个sym---->分两种以下场景
      2.1 组成syms的sym, 至少有一个来自于场景1产生 ----> 此时该输入应当被优化,转化为计算类算子产生后的结果
      TO DO (当前实现中并未处理这个场景, 离线会出问题）
      2.2 组成syms的sym, 全都来自场景三 --->暂无相关用例，如果存在，不会被优化（离线场景会出问题）
    3、用户的int\list[int]被转化为sym---->来自用户的输入，不能优化掉
      TO DO 离线场景下该如何还原,list[int]被转化为sym的场景,因为用户给的是list[int]入图的却是int
    '''

    sym_input_node_list = [] # 表示可能被删除的node列表
    # False mean delete inputs index input, True mean save inputs index input
    inputs_save_flag = []
    placeholder_nodes_list = _get_placeholder_nodes(gm)

    for node in placeholder_nodes_list:
        if is_sym(node.meta['val']):
            if node.users == {}:
                gm.graph.erase_node(node)
                inputs_save_flag.append(False)
            else:
                logger.debug(
                    f' add sym_input_node_list node: {node}, op: {node.op}, target: {node.target}'
                    +
                    f' users: {node.users}, meta: {node.meta}, type: {node.type},'
                    + f' input_nodes: {node._input_nodes}')
                sym_input_node_list.append(node)
                inputs_save_flag.append(node)  # 存入node 表示未来可能被删 也可能不被删除
        else:
            inputs_save_flag.append(True)

    logger.info(f' sym_input_node_list: {sym_input_node_list}')

    for node in placeholder_nodes_list:
        if len(sym_input_node_list) == 0:
            break
        if not is_fake(node.meta['val']): # Fake tensor not sym
            continue

        drop_sym_index_list = []
        for i, will_del_node in enumerate(sym_input_node_list):
            for j in range(len(node.meta['val'].size())):
                # we cannot use == to compare symint, because Symint will from Sx change to int
                if str(will_del_node.meta['val']) == str(node.meta['val'].size()[j]):
                    # 这里表达了，sym必须是shape的某一维度,所以能够识别场景1
                    logger.debug(
                        f' will replaced node: {will_del_node}, op: {will_del_node.op}, '
                        + f'target: {will_del_node.target}, users: {will_del_node.users}, '
                        + f'meta: {will_del_node.meta}, type: {will_del_node.type}, '
                        + f'input_nodes: {will_del_node._input_nodes}')
                    logger.debug(
                        f' inserting_after node: {node}, op: {node.op}, target: {node.target}'
                        +
                        f' users: {node.users}, meta: {node.meta}, type: {node.type},'
                        + f' input_nodes: {node._input_nodes}')
                    with gm.graph.inserting_after(node):
                        new_add_node = gm.graph.create_node(op="call_function", target=torch.ops.aten.sym_size,
                                                            args=(node, j))
                        will_del_node.replace_all_uses_with(new_add_node, propagate_meta=True)
                        logger.debug(
                            f' new_add_node: {new_add_node}, op: {new_add_node.op}, target: {new_add_node.target},'
                            +
                            f' users: {new_add_node.users}, meta: {new_add_node.meta}, type: {new_add_node.type},'
                            + f' input_nodes: {new_add_node._input_nodes}')
                    gm.graph.erase_node(will_del_node)
                    drop_sym_index_list.append(i)
                    break


        # 处理完的sym，就把他从待删除的列表中删除
        for index in reversed(drop_sym_index_list):
            del sym_input_node_list[index]

    for i in range(len(inputs_save_flag)):
        if isinstance(inputs_save_flag[i], torch.fx.node.Node):
            if inputs_save_flag[i] in sym_input_node_list:
                inputs_save_flag[i] = True
                # 这里保证真正输入的sym没有被删除，但是他不知道 sym 可能是symlist变的，所以在线ok,离线会出问题
                logger.info(f'graph has int input, sx not come from input node: {node}, op: {node.op}, '
                            + f'target: {node.target} users: {node.users}, meta: {node.meta}, type: {node.type}')
            else:
                inputs_save_flag[i] = False

    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm, inputs_save_flag


def _optimize_fx(example_inputs, gm: torch.fx.GraphModule, graph: ConcreteGraph):
    gm, input_save_list_flag = _eliminate_sym(example_inputs, gm)

    # more pass in this place

    optimize_fx_input = []
    fx_inputs_mapping = {}
    assert len(input_save_list_flag) == len(example_inputs)
    for i in range(len(input_save_list_flag)):
        if input_save_list_flag[i]:
            fx_inputs_mapping[i] = len(optimize_fx_input)
            optimize_fx_input.append(example_inputs[i])

    graph.set_fx_inputs_mapping(fx_inputs_mapping)
    logger.debug(f'after all pass graph: {gm.graph}')
    return optimize_fx_input


def _eliminate_sym(example_inputs, gm: torch.fx.GraphModule):
    dynamic = False
    input_save_list_flag = [True for input in example_inputs]
    for inp in example_inputs:
        if is_sym(inp):
            dynamic = True
            break
    if not dynamic:
        return gm, input_save_list_flag

    gm, input_save_list_flag = _remove_sym_from_shape(gm)
    assert len(input_save_list_flag) == len(example_inputs)

    logger.debug(f'after eliminate_sym graph: {gm.graph}')
    return gm, input_save_list_flag


class _NpuFxCompiler:
    def __init__(self, compiler_config: CompilerConfig) -> None:
        self.config = compiler_config

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        logger.info(f'compiler inputs')
        for i, inp in enumerate(example_inputs):
            logger.info(f'  input {i}: {inp}')
        logger.info(f'  graph: {gm.graph}')

        if self.config.debug.fx_summary.enabled:
            summarize_fx_graph(
                gm, example_inputs, self.config.debug.fx_summary.full_path("summary"))
            if self.config.debug.fx_summary.skip_compile:
                logger.warning(f'When summarizing FX Graph, npu compilation will be skipped, '
                               'and FALLBACK to EAGER execution to ensure the integrity of the analysis data. '
                               'Once the analysis is complete, please make sure to disable the summary config '
                               'to ensure that the graph is compiled and executed.')
                return gm

        with no_dispatch():
            mutable_gm = copy.deepcopy(gm)
        concrete_graph: ConcreteGraphBase = NpuGraphConverter(
            mutable_gm, graph=ConcreteGraph(self.config), garbage_collect_values=False).run(*example_inputs)

        if not self.config.export.export_mode:
            if self.config.debug.graph_dump.enabled:
                concrete_graph.dump(self.config.debug.graph_dump.full_path("dynamo_original_graph"))


        data_dumper = None
        if self.config.debug.data_dump.enabled:
            data_dumper = NpuFxDumper(gm, config=self.config.debug.data_dump)

        def inference(*args, npu_compiled_gm, original_gm, data_dumper: NpuFxDumper, **kwargs):
            with record_function("npu_fx_compiler inference"):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('runtime inputs')
                    for i, inp in enumerate(args):
                        logger.debug(f'  input {i}: {_summary(inp)}')
                    for k, v in kwargs.items():
                        logger.debug(f'  input {k}: {_summary(v)}')

                if data_dumper is not None:
                    logger.warning(f'When dumping data of FX Graph, npu run will be skipped, '
                                   'and FALLBACK to EAGER execution, once dump finished, please make sure to disable '
                                   'the data dump config to ensure that the graph is compiled and executed.')
                    compiled_result = data_dumper.run(*args, **kwargs)
                else:
                    compiled_result = npu_compiled_gm(*args, **kwargs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('runtime outputs')
                    for i, inp in enumerate(compiled_result):
                        logger.debug(f'  output {i}: {_summary(inp)}')

                return compiled_result

        return functools.partial(inference, npu_compiled_gm=concrete_graph, original_gm=gm, data_dumper=data_dumper)


def get_compiler(compiler_config: CompilerConfig = None):
    if compiler_config is None:
        compiler_config = CompilerConfig()
    return _NpuFxCompiler(compiler_config)


def get_partition_fn(compiler_config: CompilerConfig = None):
    
    def partition_fn(graph: torch.fx.GraphModule, joint_inputs, **kwargs):
        joint_graph_passes(graph)
        return default_partition(graph, joint_inputs, **kwargs)
    
    if compiler_config is None:
        compiler_config = CompilerConfig()
    if compiler_config.experimental_config.npu_fx_pass:
        return partition_fn
    return default_partition


def wrap_compiler(compiler: Callable, compiler_config: CompilerConfig):
    @functools.wraps(compiler)
    def wrapped_compiler(
        gm: torch.fx.GraphModule, 
        example_inputs: List[torch.Tensor], 
        is_inference: bool
        ):
        if is_inference and compiler_config.experimental_config.npu_fx_pass:
            joint_graph_passes(gm) 
        return compiler(gm, example_inputs)
    
    @functools.wraps(compiler)
    def joint_compiler(
        gm: torch.fx.GraphModule, 
        example_inputs: List[torch.Tensor]
        ):
        if compiler_config.experimental_config.npu_fx_pass:
            joint_graph_passes(gm) 
        return compiler(gm, example_inputs)
    
    fw_compiler = functools.partial(wrapped_compiler, is_inference=False)
    inference_compiler = functools.partial(wrapped_compiler, is_inference=True)
    return fw_compiler, inference_compiler, joint_compiler


def _npu_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor],
                 compiler_config: CompilerConfig = None, aot_config: AotConfig = None,
                 decompositions: Dict = {}):
    compiler = get_compiler(compiler_config)
    fw_compiler, inference_compiler, joint_compiler = wrap_compiler(compiler, compiler_config)
    partition_fn = get_partition_fn(compiler_config)
    if aot_config is not None and aot_config.enable_joint_graph:
        return aot_module_simplified_joint(gm, example_inputs,
                                           compiler=joint_compiler, decompositions=decompositions,
                                           output_loss_index=int(aot_config.output_loss_index.value))
    keep_inference_input_mutations = bool(compiler_config.experimental_config.keep_inference_input_mutations)
    return aot_module_simplified(gm, example_inputs, fw_compiler=fw_compiler, bw_compiler=compiler, 
                                 decompositions=decompositions, partition_fn=partition_fn, 
                                 keep_inference_input_mutations=keep_inference_input_mutations,
                                 inference_compiler=inference_compiler)


def get_npu_backend(*, compiler_config: CompilerConfig = None,
                    aot_config: AotConfig = None, custom_decompositions: Dict = {}):
    if compiler_config is None:
        compiler_config = CompilerConfig()
        
    decompositions = get_decompositions(get_default_decompositions())
    decompositions.update(custom_decompositions)
    
    add_npu_patch(decompositions)

    return functools.partial(_npu_backend, compiler_config=compiler_config, aot_config=aot_config,
                             decompositions=decompositions)