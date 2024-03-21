from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
from datetime import datetime
import functools
import threading
import contextlib
from contextlib import contextmanager
import inspect
import sys
import os
import warnings
import sympy

import torch
from torch.fx.node import Argument, Target
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor, is_fake
from torch._ops import OpOverload, OpOverloadPacket
import torch.utils._pytree as pytree

from torchair.configs.compiler_config import CompilerConfig
from torchair.core import _torchair
from torchair.core.backend import initialize_graph_engine, TorchNpuGraph
from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import Tensor as GeTensor
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type, default_ge_graph, \
    GeGraph, attr_scope, compat_as_bytes, DataType, Format, TensorSpec, is_sym, sym_to_ge_dtype
from torchair.ge_concrete_graph.graph_pass import optimize_sym_pack, optimize_reference_op_redundant_copy
from torchair.ge_concrete_graph.utils import convert_to_tensorboard, dump_graph, force_op_unknown_shape, \
    is_host_data_tensor, get_all_sym_value_mapping, get_used_syms_in_meta, Placement
from torchair.ge_concrete_graph.supported_declaration import Support
from torchair.ge_concrete_graph.export_config_generete import generate_config
from torchair.utils.export_utils import make_export_graph, get_export_file_name
from . import ge_apis as ge


def _mapping_assign_op_to_graph_output(graph: GraphDef):
    net_output: OpDef = None  # 输出节点
    net_inputs: Dict[str, int] = {}  # 输入tensor名称到索引的映射

    for op in graph.op:
        if op.type == "Data":
            net_inputs[GeTensor(op).tensor] = op.attr["index"].i
        elif op.type == "NetOutput":
            net_output = op
    assert net_output is not None, "NetOutput not found"

    def _mapping_to_graph_output(graph: GraphDef, graph_out: OpDef, assign_node_out: GeTensor, value_tensor: str):
        output_ref_index_list = []
        for i, name in enumerate(graph_out.input):
            if name == assign_node_out.tensor:
                graph_out.input[i] = value_tensor
                output_ref_index_list.append(i)
            elif name == value_tensor:
                output_ref_index_list.append(i)
        if len(output_ref_index_list) != 0 :
            return output_ref_index_list

        graph_out.input.append(value_tensor)
        graph_out.input_desc.add().CopyFrom(assign_node_out.desc)
        graph_out.input_desc[-1].name = f"input{len(graph_out.input_desc) - 1}"
        graph.attr["_output_dtypes"].list.i.append(assign_node_out.dtype)
        return [len(graph_out.input) - 1]

    output_refto_input = {}
    replaced_assign_ops = []
    for op in graph.op:
        if op.type != "Assign":
            continue
        assign_node_out = GeTensor(op)
        logger.info(f"Found assign op {assign_node_out}")
        if op.input[0] in net_inputs.keys(): # Assign在给输入赋值
            logger.info(
                f"Replace assign op {op.name} assign value from {op.input[1]} to input {net_inputs[op.input[0]]} {op.input[0]}")
            output_ref_index_list = _mapping_to_graph_output(graph, net_output, assign_node_out, op.input[1])
            for output_ref_index in output_ref_index_list:
                output_refto_input[output_ref_index] = net_inputs[op.input[0]]
            replaced_assign_ops.append(op)
        else:
            logger.info(f"Collect assign op {op.name} assign value from {op.input[1]} to {op.input[0]}")
            net_output.input.append(assign_node_out.controller)

    for op in replaced_assign_ops:
        graph.op.remove(op)

    return output_refto_input

_CONVERTERS = defaultdict(lambda: None)
_DECLARED_SUPPORTED_CONVERTERS = defaultdict(lambda: None)
_CHECKPOINT_MAP_FUNC = dict()


def _add_op_to_checkpoint_map(op, fn):
    if isinstance(op, (list, tuple)):
        for each in op:
            _add_op_to_checkpoint_map(each, fn)
        return

    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        assert isinstance(op, OpOverloadPacket)
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in _CHECKPOINT_MAP_FUNC:
            raise RuntimeError(f"duplicate registrations for checkpoint map func {op}")
        _CHECKPOINT_MAP_FUNC[op] = fn
    return


def register_checkpoint_func(ops):
    def checkpoint_decorator(fn: Callable):
        _add_op_to_checkpoint_map(ops, fn)
        return fn

    return checkpoint_decorator


def get_checkpoint_func(op: OpOverload):
    if op not in _CHECKPOINT_MAP_FUNC.keys():
        raise RuntimeError(f"Target op {op} not registered in _CHECKPOINT_MAP_FUNC.",
            "Maybe you should check if using unsupported rng ops in torch.utils.checkpoint.checkpoint.")
    return _CHECKPOINT_MAP_FUNC[op]


def _get_converter(name: Callable):
    if not name in _CONVERTERS:
        from torchair.ge_concrete_graph.ge_converter import custom
        if hasattr(name, "_ge_converter"):
            return name._ge_converter
    return _CONVERTERS[name]


def get_meta_outputs(meta_outputs):
    if isinstance(meta_outputs, (list, tuple)):
        return [get_meta_outputs(meta_output) for meta_output in meta_outputs]
    return TensorSpec(meta_outputs)


def set_ge_outputs(ge_outputs, meta_outputs):
    if isinstance(ge_outputs, ge.Tensor):
        ge_outputs.set_meta(meta_outputs)
    elif isinstance(ge_outputs, int):
        assert isinstance(meta_outputs, (torch.SymInt, int))
    else:
        if not isinstance(ge_outputs, (list, tuple)):
            raise AssertionError
        if not isinstance(meta_outputs, (list, tuple)):
            raise AssertionError
        if len(ge_outputs) != len(meta_outputs):
            raise AssertionError
        for meta_output, ge_output in zip(meta_outputs, ge_outputs):
            if meta_output is None:
                continue
            set_ge_outputs(ge_output, meta_output)


def _wrap_converter(converter: Callable):
    @functools.wraps(converter)
    def wrapped_converter(*args, **kwargs):
        meta_outputs = None
        if 'meta_outputs' in kwargs:
            meta_outputs = kwargs['meta_outputs']
            kwargs['meta_outputs'] = get_meta_outputs(meta_outputs)

        ge_outputs = converter(*args, **kwargs)

        if meta_outputs is not None:
            set_ge_outputs(ge_outputs, meta_outputs)

        return ge_outputs
    return wrapped_converter


class ExportSuccess(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f'ExportSucces: {self.message}'


class Converter:
    compile_backend = None
    result_checker = None

    def __init__(self, aten_op) -> None:
        self._aten_op = aten_op
        self._signature = inspect.signature(aten_op)
        self._supported_cases = None

    def __call__(self, converter) -> Any:
        wrapped_converter = _wrap_converter(converter)
        if 'meta_outputs' in inspect.signature(converter).parameters:
            wrapped_converter.require_meta = True
        else:
            wrapped_converter.require_meta = False
        try:
            self._aten_op._ge_converter = wrapped_converter
        except:
            global _CONVERTERS
            _CONVERTERS.update({self._aten_op: wrapped_converter})
        return self


    @property
    def supported_cases(self):
        return self._supported_cases

    @supported_cases.setter
    def supported_cases(self, supported_cases):
        for testcase in supported_cases:
            if not isinstance(testcase, Support):
                raise AssertionError
        self._supported_cases = supported_cases


def declare_supported(supported_cases: List[Support]):
    def add_testcase(converter):
        if not isinstance(converter, Converter):
            raise AssertionError
        converter.supported_cases = supported_cases
        _DECLARED_SUPPORTED_CONVERTERS.update({converter._aten_op: converter})
        return converter

    return add_testcase


def _declare_supported_converters():
    return _DECLARED_SUPPORTED_CONVERTERS


def register_fx_node_ge_converter(aten_op):
    return Converter(aten_op)


def _get_generalized_shape(fake: torch.Tensor) -> List[int]:
    generalized_shape = []
    for dim in fake.size():
        if not isinstance(dim, torch.SymInt):
            generalized_shape.append(dim)
        else:
            try:
                generalized_shape.append(int(str(dim)))
            except:
                generalized_shape.append(-1)
    return generalized_shape


def _normalize_ge_graph(graph: GraphDef):
    for op in graph.op:
        op.attr["_input_name_key"].list.s[:] = [compat_as_bytes(desc.name) for desc in op.input_desc]
        op.attr["_input_name_value"].list.i[:] = list(range(len(op.input_desc)))
        op.attr["_output_name_key"].list.s[:] = [compat_as_bytes(desc.name) for desc in op.output_desc]
        op.attr["_output_name_value"].list.i[:] = list(range(len(op.output_desc)))

        if op.type == "Data":
            continue
        for desc in op.input_desc:
            if not '_is_unfed_optional' in desc.attr:
                desc.layout = "ND"
                if desc.dtype == ProtoDataType.DT_UNDEFINED:
                    desc.dtype = ProtoDataType.DT_FLOAT
        for desc in op.output_desc:
            desc.layout = "ND"
            if desc.dtype == ProtoDataType.DT_UNDEFINED:
                desc.dtype = ProtoDataType.DT_FLOAT
        if 'input_layout_info' in op.attr:
            indices = op.attr['input_layout_info'].list.i
            layouts = op.attr['input_layout_info'].list.s
            for index, layout in zip(indices, layouts):
                desc = op.input_desc[index]
                if not '_is_unfed_optional' in desc.attr:
                    desc.layout = layout.decode()
        if 'output_layout_info' in op.attr:
            indices = op.attr['output_layout_info'].list.i
            layouts = op.attr['output_layout_info'].list.s
            for index, layout in zip(indices, layouts):
                desc = op.output_desc[index]
                desc.layout = layout.decode()


def _update_internal_format_from_inputs(graph: GraphDef, runtime_inputs):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The internal format will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip internal format updates')
        return

    torch_npu_module = sys.modules['torch_npu']
    input_index_mapping_graph_op: Dict[int, OpDef] = {}  # data节点的index到opDef的映射
    for op in graph.op:
        if op.type == "Data":
            input_index_mapping_graph_op[op.attr["index"].i] = op

    for idx in range(len(runtime_inputs)):
        if not (idx < len(input_index_mapping_graph_op)):
            raise AssertionError(
                f"GE graph input index {idx} out of Data ops index range {len(input_index_mapping_graph_op)}")

        if not runtime_inputs[idx].is_npu:
            logger.debug(f'input_{idx} is not npu tensor, skip format updates.')
            continue

        npu_format = torch_npu_module.get_npu_format(runtime_inputs[idx])
        if npu_format > Format.FORMAT_FRACTAL_Z.value \
                and npu_format != Format.FORMAT_FRACTAL_NZ.value and npu_format != Format.FORMAT_NCDHW.value:
            raise RuntimeError(f"Unsupported input tensor with format {Format(npu_format).name}.")

        # attr "format_for_int" in proto::TensorDescriptor will be be deserialized as TensorDesc Format in ge.
        input_index_mapping_graph_op[idx].output_desc[0].attr["format_for_int"].i = npu_format
        '''
           * **********************************************************************************
           *       ***********    ***********                  *************    *************
           *       *   Fmap  *    *  Filter *                  *    Fmap   *    *   Filter  *
           *       *origin:ND*    *origin:ND*                  *origin:NCHW*    *origin:NCHW*
           *       *layout:ND*    *layout:FZ*                  *layout:NCHW*    *layout:NCHW*
           *       ***********    ***********                  *************    *************
           *           \               /                            \               /
           *      ***************************                  ***************************
           *      * origin:ND  *  origin:ND *                  *origin:NCHW * origin:NCHW*
           *      *layout:NCHW * layout:NCHW*                  *layout:NCHW * layout:NCHW*
           *      ***************************        ===>      ***************************
           *             *  Conv2D   *                                *  Conv2D   *
           *             *************                                *************
           *             * origin:ND *                                *origin:NCHW*
           *             *layout:NCHW*                                *layout:NCHW*
           *             *************                                *************
           *                  |                                             |
           *              netoutput                                     netoutput
           *
           *  Figure 1. Filter's origin is ND. After GE::REfreshOriginFormatOfAnchor processing
           * ************************************************************************************
           *       ***********    *************                *************    *************
           *       *   Fmap  *    *  Filter   *                *    Fmap   *    *   Filter  *
           *       *origin:ND*    *origin:NCHW*                *origin:NCHW*    *origin:NCHW*
           *       *layout:ND*    *layout:FZ  *                *layout:NCHW*    *layout:FZ  *
           *       ***********    *************                *************    *************
           *           \               /                            \               /
           *      ***************************                  ***************************
           *      * origin:ND  *  origin:ND *                  *origin:NCHW * origin:NCHW*
           *      *layout:NCHW * layout:NCHW*                  *layout:NCHW * layout:NCHW*
           *      ***************************        ===>      ***************************
           *             *  Conv2D   *                                *  Conv2D   *
           *             *************                                *************
           *             * origin:ND *                                *origin:NCHW*
           *             *layout:NCHW*                                *layout:NCHW*
           *             *************                                *************
           *                  |                                             |
           *              netoutput                                     netoutput
           *
           *  Figure 2. Filter's origin is NCHW. After GE::REfreshOriginFormatOfAnchor processing
           * ************************************************************************************
           As shown in Figure 1, when the input of filter node origin_fmt of conv2d is ND,
           the operator cannot get the C axis. So, GE can flood storage_fmt NCHW of input1 of
           the Conv2d to filter node storage_fmt for Conv2d's constraints.
           As a result, filter node storage_fmt FZ changes to NCHW.

           Therefore, to enable the internal format FZ of the filter,
           you need to specify origin_fmt NCHW of the filter node, as shown in Figure 2.
        '''
        if npu_format == Format.FORMAT_FRACTAL_Z.value or npu_format == Format.FORMAT_NC1HWC0.value:
            input_index_mapping_graph_op[idx].output_desc[0].attr["origin_format_for_int"].i = Format.FORMAT_NCHW.value
        logger.debug(f'update the Format of output TensorDesc for input_{idx} to Format {Format(npu_format).name}.')


class ExecutorType:
    CPU = 0
    NPU = 1


def _get_executor_type():
    if 'torch_npu' in sys.modules:
        return ExecutorType.NPU
    return ExecutorType.CPU


class ViewOfInput:
    def __init__(self, index, meta_output, sym_value_mapping):
        self._fx_input_index = index
        self._sym_value_mapping = sym_value_mapping
        self._output_shape = list(meta_output.size())
        self._output_stride = list(meta_output.stride())
        self._output_offset = meta_output.storage_offset()

    def compute_output(self, *args):
        real_input = args[self._fx_input_index]
        if self._sym_value_mapping is not None:
            value_of_sym = self._compute_value_of_sym(*args)
            self._compute_output_shape_stride_offset(value_of_sym)
        return torch.as_strided(real_input, self._output_shape, self._output_stride, self._output_offset)

    def _compute_value_of_sym(self, *args):
        value_of_sym = {}
        for sym, index in self._sym_value_mapping.items():
            if index[0] == -1:
                value_of_sym[sym] = args[index[1]]
            else:
                value_of_sym[sym] = args[index[1]].size()[index[0]]
        return value_of_sym

    def _compute_output_shape_stride_offset(self, value_of_sym):
        for idx, output_shape in enumerate(self._output_shape):
            self._output_shape[idx] = sympy.simplify(str(output_shape)).subs(value_of_sym)
        for idx, output_stride in enumerate(self._output_stride):
            self._output_stride[idx] = sympy.simplify(str(output_stride)).subs(value_of_sym)
        self._output_offset = sympy.simplify(str(self._output_offset)).subs(value_of_sym)


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, graph=None, name=None):
        self._graph = GeGraph() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []
        self._fx_inputs_mapping = dict()
        self._input_placements = []
        self._frozen_flag_list = []
        self._data_index_after_forozen = dict()
        self._graph_output_ref_input = {}
        self._ref_data_idx = []
        self._executor = TorchNpuGraph(name)
        self._config = config
        self._auto_tune_times = 0
        self._converter_ctx = threading.local()
        self._fx_input_mapping_cloned_ge_input = []
        self._inputs_processing = None
        self._is_compiled = False
        self._pack_input_processing = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._config.experimental_config.frozen_parameter:
            if not self._is_compiled:
                warnings.warn(f'When enable frozen_parameter, Parameters will be considered frozen.'
                              'Please make sure that the Parameters data address remain the same '
                              'throughout the program runtime.')
                self._arg_is_frozen(*args)
                self._process_data_to_constplaceholder(*args)
                self._process_fx_inputs_mapping_and_input_placements(len(args))

        if self._pack_input_processing is None:
            self._pack_input_processing = optimize_sym_pack(self.graph, self.inputs, self._input_placements,
                                                             self._fx_inputs_mapping, len(args))
        fx_inputs = self._pack_input_processing(*args)

        if self._inputs_processing is None:
            self._inputs_processing = self._make_inputs_processing_func(*fx_inputs)
        inputs = self._inputs_processing(*fx_inputs)

        inputs = self._consume_data_into_inputs(inputs)

        self.common_graph_optimization()

        if self.config.export.export_mode:
            self.export(inputs)
            raise ExportSuccess("export graph over")

        self.load(inputs)

        if self.should_auto_tune:
            self.auto_tune(inputs)

        self.compile()

        if len(self._graph_output_ref_input):
            assigned_outputs = [None] * len(self.graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in self._graph_output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]
            ge_outputs = self._executor.run(inputs, assigned_outputs)

        else:
            ge_outputs = self._executor.run(inputs)

        if len(self._ref_data_idx) != 0:
            for index_tuple in self._fx_input_mapping_cloned_ge_input:
                if index_tuple[1] in self._ref_data_idx:
                    args[index_tuple[0]].copy_(inputs[index_tuple[1]])

        if len(ge_outputs) != len(self.graph.attr["_output_dtypes"].list.i):
            raise AssertionError(
                f"output size mismatch, expect {len(self.graph.attr['_output_dtypes'].list.i)}, got {len(ge_outputs)}")

        fx_outputs = [v for v in self._fx_outputs]
        for fx_idx, fx_output in enumerate(fx_outputs):
            if isinstance(fx_output, ViewOfInput):
                fx_outputs[fx_idx] = fx_output.compute_output(*args)

        for fx_idx, ge_idx in self._fx_outputs_mapping.items():
            if not (ge_idx < len(ge_outputs)):
                raise AssertionError(f"output index {ge_idx} out of range {len(ge_outputs)}")
            fx_outputs[fx_idx] = ge_outputs[ge_idx]

        del ge_outputs

        return tuple(fx_outputs)

    def common_graph_optimization(self) -> Any:
        if self._is_compiled:
            return

        self.graph.attr["_input_placements"].list.i.extend(self._input_placements)
        self.graph.attr["_output_dtypes"].list.i.extend([output.dtype for output in self.outputs])
        self.graph.attr["_executor_type"].i = _get_executor_type()
        self._complement_graph_attr()

        self._ref_data_idx = optimize_reference_op_redundant_copy(self.graph)
        self._graph_output_ref_input = _mapping_assign_op_to_graph_output(self.graph)

    def load(self, runtime_inputs) -> Any:
        if self._is_compiled:
            return

        # Initialize based on global options
        local_compile_options, global_compile_options = self.config.as_dict()
        global_compile_options["ge.exec.staticMemoryPolicy"] = "2"
        logger.info("global compile options:")
        for k, v in global_compile_options.items():
            logger.info(f"  {k}: {v}")

        initialize_graph_engine(global_compile_options)

        # Update local options
        output_reuse_indexes = [x for x in range(len(self.outputs)) if x not in self._graph_output_ref_input.keys()]
        if len(output_reuse_indexes) != 0:
            # support output memory reuse while output is not ref to input
            local_compile_options["ge.exec.outputReuseMemIndexes"] = ",".join(str(x) for x in output_reuse_indexes)
        local_compile_options["ge.deterministic"] = "1" if torch.are_deterministic_algorithms_enabled() else "0"
        logger.info("local compile options:")
        for k, v in local_compile_options.items():
            logger.info(f"  {k}: {v}")

        # Normalize graph
        _normalize_ge_graph(self.graph)

        _update_internal_format_from_inputs(self.graph, runtime_inputs)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path("dynamo_optimized_graph"))
        self._executor.load(self.graph, local_compile_options)

    def compile(self) -> Any:
        if self._is_compiled:
            return

        logger.info(f'start compile graph: {self}.')
        self._executor.compile()
        self._is_compiled = True
        logger.info(f'end compile graph: {self} and start run graph.')

    def context(self):
        return default_ge_graph(self.graph)

    @property
    def is_dynamic(self):
        if hasattr(self, '_is_dynamic'):
            return self._is_dynamic

        for input in self._inputs:
            if (-1 in input.desc.shape.dim) or (-2 in input.desc.shape.dim):
                self._is_dynamic = True
                return True

        self._is_dynamic = False
        return False

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if is_sym(meta_outputs):
            data = ge.Data(index=len(self._inputs),
                           dtype=sym_to_ge_dtype(meta_outputs), shape=[], placement='CPU', node_name=target)
            data.set_meta(meta_outputs)
            self._inputs.append(data)
            self._input_placements.append(Placement.HOST)
        else:
            if not isinstance(meta_outputs, torch.Tensor):
                raise AssertionError
            dtype = torch_type_to_ge_type(meta_outputs.dtype)
            shape = _get_generalized_shape(meta_outputs)
            placement = Placement.HOST if (
                meta_outputs.device is None or meta_outputs.device.type == 'cpu') else Placement.DEVICE
            data = ge.Data(index=len(self._inputs), dtype=dtype,
                           shape=shape, placement='CPU' if (placement == Placement.HOST) else 'NPU', node_name=target)
            data.set_meta(meta_outputs)
            self._inputs.append(data)
            self._input_placements.append(placement)
        return self._inputs[-1]

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise AssertionError
        args = args[0]
        fx_inputs_mapping_reverse = {value : key for key, value in self._fx_inputs_mapping.items()}
        all_sym_value_mapping = get_all_sym_value_mapping(fx_inputs_mapping_reverse, self.inputs)

        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            self._fx_outputs.append(arg)
            if not isinstance(arg, ge.Tensor):
                continue

            is_view2output_flag = False
            for input_index, input in enumerate(self.inputs):
                if torch._C._is_alias_of(input.meta, arg.meta):
                    is_view2output_flag = True
                    fx_input_index = fx_inputs_mapping_reverse[input_index]
                    used_syms_in_meta_output = get_used_syms_in_meta(arg.meta)
                    sym_value_mapping = {sym: all_sym_value_mapping[str(sym)] for sym in used_syms_in_meta_output}
                    self._fx_outputs[-1] = ViewOfInput(fx_input_index, arg.meta, sym_value_mapping)
                    break
            if is_view2output_flag:
                continue

            output_index = len(self._outputs)
            for i, output in enumerate(self._outputs):
                if output.tensor == arg.tensor:
                    output_index = i
                    break
            self._fx_outputs_mapping[len(
                self._fx_outputs) - 1] = output_index
            if output_index == len(self._outputs):
                self._outputs.append(arg)

        ge.NetOutput(self._outputs)
        return args

    def parse_symlist(self, syms):
        npu_syms = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                npu_syms.append(sym.npu)
            else:
                if not isinstance(sym, int):
                    raise AssertionError
                npu_syms.append(sym)
        if all([isinstance(sym, int) for sym in npu_syms]):
            return npu_syms

        pack_tensor = ge.Pack(npu_syms, N=len(npu_syms), axis=0)
        if all([is_host_data_tensor(sym_i) for sym_i in npu_syms]):
            pack_tensor.node.attr['_inputs_all_sym'].b = True

        # force unknown shape with ge.Pack when parse symlist
        return force_op_unknown_shape(pack_tensor)

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if hasattr(target, "_ge_converter"):
            converter = target._ge_converter
        else:
            converter = _get_converter(target)
        if converter is None:
            raise RuntimeError(f"Unsupported torch op {target} by ge")
        if converter.require_meta:
            ge_outputs = converter(*args, **kwargs, meta_outputs=meta_outputs)
            if meta_outputs is not None and hasattr(self._converter_ctx, 'node') and self._converter_ctx.node:
                fx_tensor_prefix = f'{self._converter_ctx.node.name}-{self._converter_ctx.node.target}.OUTPUT'
                if isinstance(ge_outputs, ge.Tensor):
                    ge_outputs.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.0')
                elif isinstance(ge_outputs, (list, tuple)) and all([isinstance(v, ge.Tensor) for v in ge_outputs]):
                    for i, ge_output in enumerate(ge_outputs):
                        ge_output.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.{i}')
            return ge_outputs
        else:
            return converter(*args, **kwargs)

    def dump(self, path: str):
        dump_graph(path, self.graph)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    @property
    def config(self):
        return self._config

    def set_fx_inputs_mapping(self, fx_inputs_mapping):
        self._fx_inputs_mapping = fx_inputs_mapping

    def export(self, inputs) -> Any:
        file_path = self.config.export.export_path_dir
        file_name_air = get_export_file_name(self.config.export.export_name)

        export_graph = make_export_graph(inputs, self.config, self.graph)

        _normalize_ge_graph(export_graph)

        if self.config.export.experimental.auto_atc_config_generated:
            generate_config(self.config, file_path, export_graph)
        local_options = {}
        local_options["export_path_dir"] = file_path
        local_options["export_name"] = file_name_air
        _torchair.export(export_graph.SerializeToString(), local_options)

    @property
    def should_auto_tune(self) -> bool:
        if self.config.aoe_config.aoe_mode.value is None:
            return False
        if self.is_dynamic:
            # AOE is not supported for dynamic shape now
            return False
        return self._auto_tune_times == 0

    @contextmanager
    def converter_context(self, *, node):
        try:
            self._converter_ctx.node = node
            attr_maps = {}
            if self.config.export.experimental.enable_record_nn_module_stack:
                stack = node.meta.get("nn_module_stack")
                if stack is not None:
                    attr_maps["nn_module_stack"] = compat_as_bytes(str(stack))
            with attr_scope(attr_maps):
                yield
        finally:
            self._converter_ctx.node = None

    def auto_tune(self, inputs: List[Tensor]) -> Any:
        logger.info(f"Start auto tune for round {self._auto_tune_times}")
        self._executor.auto_tune(inputs)
        self._auto_tune_times += 1
        logger.info(f"End auto tune for round {self._auto_tune_times - 1}")

    def _complement_graph_attr(self):
        num_inputs = self.graph.num_inputs()
        diff = num_inputs - len(self._inputs)
        self._inputs.extend([None for _ in range(diff)])
        self._input_placements.extend([None for _ in range(diff)])
        self.graph.attr["_input_placements"].list.i.extend(
            [-1 for _ in range(diff)])

        for gen in self.graph.generator_rng_state:
            rng_state = self.graph.get_graph_rng_state(gen)
            idx, offset_data = rng_state.get_idx_and_offset()
            if len(self._inputs) != len(self._input_placements):
                raise AssertionError
            placement = Placement.HOST
            self._inputs[idx] = offset_data
            self._input_placements[idx] = placement
            self.graph.attr["_input_placements"].list.i[idx] = placement

    def _consume_data_into_inputs(self, inputs):
        num_inputs = self.graph.num_inputs()
        diff = num_inputs - len(inputs)
        if diff > 0:
            istuple = isinstance(inputs, tuple)
            inputs = list(inputs)
            inputs.extend([None for _ in range(diff)])
            for gen in self.graph.generator_rng_state:
                rng_state = self.graph.get_graph_rng_state(gen)
                idx, offset = rng_state.consume()
                inputs[idx] = offset
            inputs = tuple(inputs) if istuple else inputs
        return inputs

    def _make_inputs_processing_func(self, *args: Any):
        uncontiguous_ge_input_idx = []
        nontensor_ge_input_idx = []
        for fx_index, ge_index in self._fx_inputs_mapping.items():
            if isinstance(args[fx_index], torch.Tensor):
                if not args[fx_index].is_contiguous():
                    uncontiguous_ge_input_idx.append(ge_index)
                    self._fx_input_mapping_cloned_ge_input.append((fx_index, ge_index))
            else:
                nontensor_ge_input_idx.append(ge_index)

        if len(self._fx_inputs_mapping) == len(args):
            if len(uncontiguous_ge_input_idx) == 0 and len(nontensor_ge_input_idx) == 0:
                def inputs_processing(*args: Any):
                    return args
            else:
                def inputs_processing(*args: Any):
                    inputs = list(args)
                    for idx in uncontiguous_ge_input_idx:
                        inputs[idx] = inputs[idx].contiguous()
                    for idx in nontensor_ge_input_idx:
                        inputs[idx] = torch.tensor(inputs[idx])
                    return inputs
        else:
            def inputs_processing(*args: Any):
                inputs = [None] * len(self._fx_inputs_mapping)
                for fx_idx, ge_idx in self._fx_inputs_mapping.items():
                    inputs[ge_idx] = args[fx_idx]

                for idx in uncontiguous_ge_input_idx:
                    inputs[idx] = inputs[idx].contiguous()
                for idx in nontensor_ge_input_idx:
                    inputs[idx] = torch.tensor(inputs[idx])
                return inputs

        return inputs_processing

    def _arg_is_frozen(self, *args: Any):
        data_num = 0
        for idx, arg in enumerate(args):
            if isinstance(arg, torch.nn.Parameter):
                self._frozen_flag_list.append(True)
                logger.info(f"No.{idx} arg is ConstPlaceHolder")
            else:
                self._frozen_flag_list.append(False)
                logger.info(f"No.{idx} arg is Data")
                if idx in self._fx_inputs_mapping:
                    self._data_index_after_forozen[idx] = data_num
                    data_num += 1

    def _process_fx_inputs_mapping_and_input_placements(self, args_len):
        new_input_placements = []
        new_inputs = []
        new_graph_indexed_inputs = {}
        for idx in range(args_len):
            if idx in self._fx_inputs_mapping:
                if not self._frozen_flag_list[idx]:
                    new_input_placements.append(self._input_placements[self._fx_inputs_mapping[idx]])
                    new_inputs.append(self._inputs[self._fx_inputs_mapping[idx]])
                    new_graph_indexed_inputs[self._data_index_after_forozen[idx]] = \
                        self.graph._indexed_inputs[self._fx_inputs_mapping[idx]]
        self._inputs = new_inputs
        self._input_placements = new_input_placements
        self.graph._indexed_inputs = new_graph_indexed_inputs
        self._fx_inputs_mapping = self._data_index_after_forozen

    def _process_data_to_constplaceholder(self, *args: Any):
        name_mapping_data_to_constplaceholder = dict()
        fx_inputs_mapping_reverse = {value : key for key, value in self._fx_inputs_mapping.items()}
        frozen_data_op_list = []
        for op in self.graph.op:
            if op.type == "Data":
                args_index = fx_inputs_mapping_reverse[op.attr["index"].i]
                if self._frozen_flag_list[args_index]:
                    frozen_data_op_list.append(op)
        for op in frozen_data_op_list:
            args_index = fx_inputs_mapping_reverse[op.attr["index"].i]
            name = f"ConstPlaceHolder_{args_index}_{args[args_index].data_ptr()}"
            name_mapping_data_to_constplaceholder[op.name] = name
            origin_shape = _get_generalized_shape(args[args_index])
            dtype = torch_type_to_ge_type(args[args_index].dtype)
            addr = args[args_index].data_ptr()
            with self.graph:
                if 'torch_npu' in sys.modules:
                    _torch_npu_module = sys.modules['torch_npu']
                    from torchair.core import _npu_graph_executor as _npu_executor
                    storage_format = _torch_npu_module.get_npu_format(args[args_index])

                    constplaceholder = ge.ConstPlaceHolder(
                        origin_shape=origin_shape,
                        origin_format=2 if len(origin_shape) != 4 else 0,
                        storage_shape=_npu_executor.GetNpuStorageSizes(args[args_index]),
                        storage_format=storage_format,
                        expand_dim_rules="", dtype=dtype, addr=addr,
                        size=_torch_npu_module.get_storage_size(args[args_index]) * args[args_index].element_size(),
                        node_name=name)
                    logger.debug(f'Construct ConstPlaceHolder_{op.attr["index"].i} from npu tensor, '
                                 f'storage format={Format(storage_format).name}, '
                                 f'storage shape={_npu_executor.GetNpuStorageSizes(args[args_index])}.')
                else:
                    constplaceholder = ge.ConstPlaceHolder(
                        origin_shape=origin_shape,
                        origin_format=2,
                        storage_shape=origin_shape,
                        storage_format=2,
                        expand_dim_rules="", dtype=dtype, addr=addr,
                        size=args[args_index].numel() * args[args_index].element_size(),
                        node_name=name)
                    logger.debug(f'Construct ConstPlaceHolder_{op.attr["index"].i} from cpu tensor, '
                                 f'storage format=FORMAT_ND, '
                                 f'storage shape={origin_shape}.')
                constplaceholder.set_meta(self.inputs[op.attr["index"].i]._meta)
        # 删除是constplaceholder的Data节点
        for frozen_data_op in frozen_data_op_list:
            for i, op in enumerate(self.graph.op):
                if op.type == "Data" and op.attr["index"].i == frozen_data_op.attr["index"].i:
                    del self.graph.op[i]
        # 修改不是constplaceholder的Data节点的index
        for op in self.graph.op:
            if op.type == "Data":
                args_index = fx_inputs_mapping_reverse[op.attr["index"].i]
                self._assert_args_checkout(args_index in self._data_index_after_forozen)
                op.attr["index"].i = self._data_index_after_forozen[args_index]
        # 处理边
        for op in self.graph.op:
            for idx, op_input in enumerate(op.input):
                for key in name_mapping_data_to_constplaceholder.keys():
                    if key in op_input:
                        op.input[idx] = f"{name_mapping_data_to_constplaceholder[key]}:{op.input[idx][-1]}"
                        break

    def _assert_args_checkout(arg_flag):
        if not arg_flag:
            raise AssertionError
