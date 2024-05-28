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
from torchair.core._backend import initialize_graph_engine
from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import Tensor as GeTensor
from torchair.ge_concrete_graph.ge_graph import _ValueInput, _TensorInput, _DiscontiguousTensorInput, _RngStatusInput
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type, default_ge_graph, \
    GeGraph, attr_scope, compat_as_bytes, DataType, Format, TensorSpec, is_sym, sym_to_ge_dtype, assert_args_checkout
from torchair.ge_concrete_graph.graph_pass import optimize_sym_pack, optimize_reference_op_redundant_copy, \
    replace_data_to_refdata, get_frozen_flag, frozen_data_by_constplaceholder
from torchair.ge_concrete_graph.utils import convert_to_tensorboard, dump_graph, force_op_unknown_shape, \
    is_host_data_tensor, get_used_sym_value_mapping, Placement, compute_value_of_sym, \
    generate_sym_exper, get_sym_int_value, generate_shape_from_tensor, update_op_input_name_from_mapping
from torchair.ge_concrete_graph.supported_declaration import Support
from torchair.ge_concrete_graph.continguous_utils import guard_view_input
from torchair.ge_concrete_graph.export_config_generete import generate_config
from torchair._utils.export_utils import make_export_graph, get_export_file_name
from . import ge_apis as ge


def _mapping_assign_op_to_graph_output(graph: GraphDef):
    net_output: OpDef = None  # 输出节点
    net_inputs: Dict[str, int] = {}  # 输入tensor名称到索引的映射

    for op in graph.op:
        if op.type == "Data":
            net_inputs[GeTensor(op).tensor] = op.attr["index"].i
        elif op.type == "NetOutput":
            net_output = op
    if not net_output:
        raise AssertionError("NetOutput not found")

    def _mapping_to_graph_output(graph: GraphDef, graph_out: OpDef, assign_node_out: GeTensor, value_tensor: str):
        output_ref_index_list = []
        for i, name in enumerate(graph_out.input):
            if name == assign_node_out.tensor:
                graph_out.input[i] = value_tensor
                output_ref_index_list.append(i)
            elif name == value_tensor:
                output_ref_index_list.append(i)
        if len(output_ref_index_list) != 0:
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
        if op.input[0] in net_inputs.keys():  # Assign在给输入赋值
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
_SUPPORT_FORMAT_SET = {
    Format.FORMAT_UNDEFINED.value,
    Format.FORMAT_NCHW.value,
    Format.FORMAT_NHWC.value,
    Format.FORMAT_ND.value,
    Format.FORMAT_NC1HWC0.value,
    Format.FORMAT_FRACTAL_Z.value,
    Format.FORMAT_FRACTAL_NZ.value,
    Format.FORMAT_NCDHW.value,
    Format.FORMAT_NDC1HWC0.value,
    Format.FORMAT_FRACTAL_Z_3D.value
}


def _add_op_to_checkpoint_map(op, fn):
    if isinstance(op, (list, tuple)):
        for each in op:
            _add_op_to_checkpoint_map(each, fn)
        return

    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        if not isinstance(op, OpOverloadPacket):
            raise AssertionError("op must be an instance of OpOverloadPacket.")
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
        if not isinstance(meta_outputs, (torch.SymInt, int)):
            raise AssertionError("meta_outputs must be a torch.SymInt or an integer.")
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


def empty_function(*args, **kwargs):
    pass


def register_fx_node_ge_converter(aten_op):
    if aten_op is None:
        return empty_function
    return Converter(aten_op)


def _normalize_ge_graph(graph: GraphDef):
    for op in graph.op:
        op.attr["_input_name_key"].list.s[:] = [compat_as_bytes(desc.name) for desc in op.input_desc]
        op.attr["_input_name_value"].list.i[:] = list(range(len(op.input_desc)))
        op.attr["_output_name_key"].list.s[:] = [compat_as_bytes(desc.name) for desc in op.output_desc]
        op.attr["_output_name_value"].list.i[:] = list(range(len(op.output_desc)))

        if op.type == "Data" or op.type == "RefData":
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
        if op.type == "Data" or op.type == "RefData":
            input_index_mapping_graph_op[op.attr["index"].i] = op

    for idx in range(len(runtime_inputs)):
        if not (idx < len(input_index_mapping_graph_op)):
            raise AssertionError(
                f"GE graph input index {idx} out of Data ops index range {len(input_index_mapping_graph_op)}")

        if not runtime_inputs[idx].is_npu:
            logger.debug(f'input_{idx} is not npu tensor, skip format updates.')
            continue

        npu_format = torch_npu_module.get_npu_format(runtime_inputs[idx])
        if npu_format not in _SUPPORT_FORMAT_SET:
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
        if npu_format == Format.FORMAT_FRACTAL_Z_3D.value or npu_format == Format.FORMAT_NDC1HWC0.value:
            input_index_mapping_graph_op[idx].output_desc[0].attr["origin_format_for_int"].i = Format.FORMAT_NCDHW.value
        logger.debug(f'update the Format of output TensorDesc for input_{idx} to Format {Format(npu_format).name}.')


def _update_constplaceholder_attr_from_inputs(graph: GraphDef, runtime_inputs):
    update_node_name_mapping = {}
    for op in graph.op:
        if op.type == "ConstPlaceHolder":
            if not 'update_node_from_fx_input_idx' in op.attr:
                logger.debug(f'No need to update {op.name}.')
                continue
            real_input = runtime_inputs[op.attr["update_node_from_fx_input_idx"].i]
            origin_shape = real_input.shape
            op.attr["origin_shape"].list.i.extend(origin_shape)
            op.attr["dtype"].dt = torch_type_to_ge_type(real_input.dtype)
            op.attr["addr"].i = real_input.data_ptr()
            if 'torch_npu' in sys.modules:
                _torch_npu_module = sys.modules['torch_npu']
                from torchair.core import _npu_graph_executor as _npu_executor
                op.attr["origin_format"].i = 2 if len(origin_shape) != 4 else 0
                op.attr["storage_shape"].list.i.extend(_npu_executor.GetNpuStorageSizes(real_input))
                op.attr["storage_format"].i = _torch_npu_module.get_npu_format(real_input)
                op.attr["size"].i = _torch_npu_module.get_storage_size(real_input) * real_input.element_size()
            else:
                op.attr["origin_format"].i = 2
                op.attr["storage_shape"].list.i.extend(origin_shape)
                op.attr["storage_format"].i = 2
                op.attr["size"].i = real_input.numel() * real_input.element_size()
            logger.debug(f'Update {op.name} attr from input {op.attr["update_node_from_fx_input_idx"].i}, '
                         f'origin shape={origin_shape}, origin format={Format(op.attr["origin_format"].i).name}, '
                         f'storage shape={op.attr["storage_shape"].list.i}, '
                         f'storage format={Format(op.attr["storage_format"].i).name}, data size={op.attr["size"].i}.')

            # Note: Cannot have multiple ConstPlaceHolder nodes with the same name and different add.
            unique_addr_name = f"{op.name}_{real_input.data_ptr()}"
            update_node_name_mapping[op.name] = unique_addr_name
            op.name = unique_addr_name
    update_op_input_name_from_mapping(graph, update_node_name_mapping)


class ExecutorType:
    CPU = 0
    NPU = 1


def _get_executor_type():
    if 'torch_npu' in sys.modules:
        return ExecutorType.NPU
    return ExecutorType.CPU


class SymOutput:
    def __init__(self, meta_output, sym_value_mapping):
        self._sym_value_mapping = sym_value_mapping
        self._ori_meta_sym = meta_output
        # 获取sym对应的sympy属性,sym对象无法存入map,sym_value_mapping和符号都使用sympy对象
        self._fx_output_smy = generate_sym_exper(meta_output)

    def compute_output(self, *args):
        value_of_sym = compute_value_of_sym(self._sym_value_mapping, *args)
        return get_sym_int_value(self._fx_output_smy, value_of_sym)


class ViewOfInput:
    def __init__(self, index, meta_output, sym_value_mapping):
        self._fx_input_index = index
        self._sym_value_mapping = sym_value_mapping
        self._ori_meta_shape = list(meta_output.size())
        self._ori_meta_stride = list(meta_output.stride())
        self._ori_meta_offset = meta_output.storage_offset()
        self._meta_output_shape = generate_sym_exper(list(meta_output.size()))
        self._meta_output_stride = generate_sym_exper(list(meta_output.stride()))
        self._meta_output_offset = generate_sym_exper(meta_output.storage_offset())


    def compute_output(self, *args):
        real_input = args[self._fx_input_index]
        value_of_sym = compute_value_of_sym(self._sym_value_mapping, *args)
        output_shape, output_stride, output_offset \
            = self._compute_output_shape_stride_offset(value_of_sym)
        return torch.as_strided(real_input, output_shape, output_stride, output_offset)


    def _compute_output_shape_stride_offset(self, value_of_sym):
        if value_of_sym is None or len(value_of_sym) == 0:
            return self._meta_output_shape, self._meta_output_stride, self._meta_output_offset
        output_shape, output_stride = [], []
        for meta_output_shape in self._meta_output_shape:
            output_shape.append(get_sym_int_value(meta_output_shape, value_of_sym))
        for meta_output_stride in self._meta_output_stride:
            output_stride.append(get_sym_int_value(meta_output_stride, value_of_sym))
        output_offset = get_sym_int_value(self._meta_output_offset, value_of_sym)
        return output_shape, output_stride, output_offset


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, name=None):
        self._graph = GeGraph(name=name)
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._fx_input_names = []
        self._input_process = None
        self._input_func_list = []
        self._graph_output_ref_input = {}
        self._ref_data_idx = []
        self._cloned_ge_input_mapping = {}
        self._config = config
        self._auto_tune_times = 0
        self._converter_ctx = threading.local()
        self._is_compiled = False
        self._all_sym_input_idx = {}
        self._all_meta_tensor_input = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._is_compiled:
            # Equivalent functionality to 'self._input_func_list[i](*args)', but better performance.
            self._input_process = self._gen_input_process(self._fx_input_names, self._all_sym_input_idx,
                                                          self._input_func_list)
        inputs = self._input_process(*args)

        self.update_graph_with_runtime(inputs, args)

        if self.config.export.export_mode:
            self.export(inputs)
            raise ExportSuccess("export graph over")

        if not self._is_compiled:
            local_compile_options, global_compile_options = self._normalize_ge_option()
            initialize_graph_engine(global_compile_options)
            self.graph.load(local_compile_options)

        if self.should_auto_tune:
            self.auto_tune(inputs)

        self.compile()

        if len(self._graph_output_ref_input):
            assigned_outputs = [None] * len(self.graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in self._graph_output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]
            ge_outputs = self.graph.run(inputs, assigned_outputs)
        else:
            ge_outputs = self.graph.run(inputs)

        if len(self._ref_data_idx) != 0:
            for ge_index, fx_index in self._cloned_ge_input_mapping.items():
                if ge_index in self._ref_data_idx:
                    args[fx_index].copy_(inputs[ge_index])

        if len(ge_outputs) != len(self.graph.attr["_output_dtypes"].list.i):
            raise AssertionError(
                f"output size mismatch, expect {len(self.graph.attr['_output_dtypes'].list.i)}, got {len(ge_outputs)}")

        fx_outputs = [v for v in self._fx_outputs]
        for fx_idx, fx_output in enumerate(fx_outputs):
            if isinstance(fx_output, ViewOfInput) or isinstance(fx_output, SymOutput):
                fx_outputs[fx_idx] = fx_output.compute_output(*args)

        for fx_idx, ge_idx in self._fx_outputs_mapping.items():
            if not (ge_idx < len(ge_outputs)):
                raise AssertionError(f"output index {ge_idx} out of range {len(ge_outputs)}")
            fx_outputs[fx_idx] = ge_outputs[ge_idx]

        del ge_outputs
        return tuple(fx_outputs)

    def optimize_graph_without_runtime(self, *args):
        from torchair.ge_concrete_graph.graph_pass import remove_dead_data_and_reorder_data_index
        from torchair.ge_concrete_graph.utils import get_graph_input_placements
        if self._config.experimental_config.frozen_parameter:
            warnings.warn(f'When enable frozen_parameter, Parameters will be considered frozen.'
                          'Please make sure that the Parameters data address remain the same '
                          'throughout the program runtime.')
            frozen_flag_list = get_frozen_flag(*args)
            frozen_data_by_constplaceholder(self.graph, frozen_flag_list, self._all_meta_tensor_input)

        optimize_sym_pack(self.graph)

        # Note:
        # Please do not take any actions to add or delete data nodes, or change the index of data nodes after this.
        self._input_func_list = remove_dead_data_and_reorder_data_index(self.graph)

        self.graph.attr["_input_placements"].list.i.extend(get_graph_input_placements(self.graph))
        self.graph.attr["_output_dtypes"].list.i.extend([output.dtype for output in self.outputs])
        self.graph.attr["_executor_type"].i = _get_executor_type()

        self._ref_data_idx = optimize_reference_op_redundant_copy(self.graph)
        self._graph_output_ref_input = _mapping_assign_op_to_graph_output(self.graph)
        for ge_index, input_func in enumerate(self._input_func_list):
            if isinstance(input_func, _DiscontiguousTensorInput):
                self._cloned_ge_input_mapping[ge_index] = input_func.fx_input_idx

        _normalize_ge_graph(self.graph)

    def update_graph_with_runtime(self, inputs, fx_inputs):
        if self._is_compiled:
            return

        _update_constplaceholder_attr_from_inputs(self.graph, fx_inputs)

        _update_internal_format_from_inputs(self.graph, inputs)

        # replace ge.Data to ge.RefData when ref input
        if self._config.experimental_config.enable_ref_data:
            ref_data_idx = set()
            for idx in self._ref_data_idx:
                ref_data_idx.add(idx)
            for k, v in self._graph_output_ref_input.items():
                ref_data_idx.add(v)
            replace_data_to_refdata(self.graph, ref_data_idx, inputs)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path("dynamo_optimized_graph"))

    def compile(self) -> Any:
        if self._is_compiled:
            return

        logger.info(f'start compile graph: {self.graph.name}.')
        self.graph.compile()
        self._is_compiled = True
        logger.info(f'end compile graph: {self.graph.name} and start run graph.')

    def codegen(self):
        from torch._inductor.utils import IndentedBuffer
        if self._config.experimental_config.enable_ref_data or self.config.export.export_mode \
                or self.config.aoe_config.aoe_mode.value is not None:
            logger.info(f'Unsupported codegen for graph with config enable_ref_data or export_mode or aoe_mode.')
            return None

        head = IndentedBuffer()
        head.splice('''
        import torch
        from torchair.core._backend import initialize_graph_engine
        from torchair.ge_concrete_graph.ge_graph import GeGraph
        from torchair.ge_concrete_graph.fx2ge_converter import _update_constplaceholder_attr_from_inputs
        from torchair.ge_concrete_graph.fx2ge_converter import _update_internal_format_from_inputs
        ''')
        head.writelines(['', f'serialized_graph = {self.graph.SerializeToString()}'])
        head.writelines(['', f'global_compile_options = {{}}'])
        local_compile_options, global_compile_options = self._normalize_ge_option()
        for k, v in global_compile_options.items():
            head.writeline(f'global_compile_options["{k}"] = "{v}"')
        head.writeline(f'local_compile_options = {{}}')
        for k, v in local_compile_options.items():
            head.writeline(f'local_compile_options["{k}"] = "{v}"')
        head.writelines(['', 'initialize_graph_engine(global_compile_options)',
                         'ge_graph = GeGraph(serialized_model_def=serialized_graph)'])

        kernel = IndentedBuffer()
        kernel.writelines(['', '_is_first_run = True', f'def kernel(*args):'])
        with kernel.indent():
            input_code = self._codegen_input(self._fx_input_names, self._all_sym_input_idx, self._input_func_list)
            kernel.splice(input_code)

            # for info update in first run
            kernel.writelines(['', 'global _is_first_run', 'if _is_first_run:'])
            with kernel.indent():
                kernel.writelines(['_is_first_run = False',
                                   '_update_constplaceholder_attr_from_inputs(ge_graph, args)',
                                   '_update_internal_format_from_inputs(ge_graph, ge_inputs)',
                                   'ge_graph.load(local_compile_options, create_pg=True)',
                                   'ge_graph.compile()'])

            kernel.writeline('')
            if len(self._graph_output_ref_input):
                kernel.writeline(f'assigned_outputs = [None] * {len(self.graph.attr["_output_dtypes"].list.i)}')
                for output_index, input_index in self._graph_output_ref_input.items():
                    kernel.writeline(f'assigned_outputs[{output_index}] = ge_inputs[{input_index}]')
                kernel.writeline(f'ge_outputs = ge_graph.run(ge_inputs, assigned_outputs)')
            else:
                kernel.writeline(f'ge_outputs = ge_graph.run(ge_inputs)')
            if len(self._ref_data_idx) != 0:
                for ge_index, fx_index in self._cloned_ge_input_mapping.items():
                    if ge_index in self._ref_data_idx:
                        kernel.writeline(f'arg{fx_index}_1.copy_(ge_inputs[{ge_index}])')

            kernel.writeline(f'fx_outputs = [None] * {len(self._fx_outputs)}')
            for idx, out in enumerate(self._fx_outputs):
                if isinstance(out, ViewOfInput):
                    kernel.writeline(f'fx_outputs[{idx}] = torch.as_strided(arg{out._fx_input_index}_1, {out._ori_meta_shape}, {out._ori_meta_stride}, {out._ori_meta_offset})')
                elif isinstance(out, SymOutput):
                    kernel.writeline(f'fx_outputs[{idx}] = {str(out._ori_meta_sym.node.expr)}')
                elif not isinstance(out, GeTensor):
                    kernel.writeline(f'fx_outputs[{idx}] = {self._fx_outputs[idx]}')
                else:
                    if idx not in self._fx_outputs_mapping.keys():
                        raise AssertionError
                    kernel.writeline(f'fx_outputs[{idx}] = ge_outputs[{self._fx_outputs_mapping[idx]}]')
            kernel.writelines(['', 'del ge_outputs', 'return tuple(fx_outputs)'])

        code = IndentedBuffer()
        code.splice(head)
        code.splice(kernel)
        return code.getvalue()

    def context(self):
        return default_ge_graph(self.graph)

    @property
    def is_dynamic(self):
        if hasattr(self, '_is_dynamic'):
            return self._is_dynamic

        for data_op in self.graph.op:
            if data_op.type != "Data":
                continue
            if (-1 in data_op.output_desc[0].shape.dim) or (-2 in data_op.output_desc[0].shape.dim):
                self._is_dynamic = True
                return True

        self._is_dynamic = False
        return False

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        data_index = self.graph.num_inputs
        self._fx_input_names.append(target)
        if is_sym(meta_outputs):
            self._all_sym_input_idx[(meta_outputs).node.expr] = data_index
            data = ge.Data(index=data_index, dtype=sym_to_ge_dtype(meta_outputs), shape=[], placement='CPU',
                           node_name=target)
            data.set_meta(meta_outputs)
            self.graph.record_input_func(data.node.name, _ValueInput(data_index))
        else:
            if not isinstance(meta_outputs, torch.Tensor):
                raise AssertionError
            self._all_meta_tensor_input[data_index] = meta_outputs
            dtype = torch_type_to_ge_type(meta_outputs.dtype)
            shape = generate_shape_from_tensor(meta_outputs)
            placement = 'CPU' if (meta_outputs.device is None or meta_outputs.device.type == 'cpu') else 'NPU'
            data = ge.Data(index=data_index, dtype=dtype, shape=shape, placement=placement, node_name=target)
            data.set_meta(meta_outputs)
            self.graph.record_input_func(data.node.name, _TensorInput(
                data_index) if meta_outputs.is_contiguous() else _DiscontiguousTensorInput(data_index))
        return data

    @guard_view_input
    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise AssertionError
        args = args[0]
        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            self._fx_outputs.append(arg)
            if not isinstance(arg, ge.Tensor):
                continue

            if is_sym(arg.meta):
                self._fx_outputs[-1] = SymOutput(arg.meta,
                                                 get_used_sym_value_mapping(self._all_sym_input_idx, arg.meta))
                continue

            is_view2output_flag = False
            for fx_input_idx, fx_input_meta in self._all_meta_tensor_input.items():
                if torch._C._is_alias_of(fx_input_meta, arg.meta):
                    self._fx_outputs[-1] = ViewOfInput(fx_input_idx, arg.meta,
                                                       get_used_sym_value_mapping(self._all_sym_input_idx, arg.meta))
                    is_view2output_flag = True
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
        pack_tensor.set_meta([sym.meta if isinstance(sym, ValuePack) else sym for sym in syms])
        if all([is_host_data_tensor(sym_i) for sym_i in npu_syms]):
            pack_tensor.node.attr['_inputs_all_sym'].b = True

        # force unknown shape with ge.Pack when parse symlist
        return force_op_unknown_shape(pack_tensor)

    @guard_view_input
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
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    @property
    def config(self):
        return self._config

    def export(self, inputs) -> Any:
        file_path = self.config.export.export_path_dir
        file_name_air = get_export_file_name(self.config.export.export_name)

        export_graph = make_export_graph(self.graph, inputs, self.config.export.export_path_dir,
                                         self.config.export.weight_name)


        _normalize_ge_graph(export_graph)
        if self.config.export.enable_save_load_mode:
            if not self._check_support_for_save_graph():
                raise RuntimeError("Check support for save graph error")
            logger.debug(f"Serialize model with save mode, graph name is {export_graph.name}")
            self._normalize_exportd_graph_attr(export_graph.model)
            serialize_save_graph(export_graph, file_path + "/" + file_name_air)
        else:
            if self.config.export.experimental.auto_atc_config_generated:
                generate_config(self.config, file_path, export_graph)
            local_options = {"export_path_dir": file_path, "export_name": file_name_air}
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
        self.graph.auto_tune(inputs)
        self._auto_tune_times += 1
        logger.info(f"End auto tune for round {self._auto_tune_times - 1}")

    def _normalize_ge_option(self):
        # Initialize based on global options
        local_compile_options, global_compile_options = self.config.as_dict()
        global_compile_options["ge.exec.staticMemoryPolicy"] = "2"
        logger.info("global compile options:")
        for k, v in global_compile_options.items():
            logger.info(f"  {k}: {v}")

        # Update local options
        output_reuse_indexes = [x for x in range(len(self.outputs)) if x not in self._graph_output_ref_input.keys()]
        if len(output_reuse_indexes) != 0:
            # support output memory reuse while output is not ref to input
            local_compile_options["ge.exec.outputReuseMemIndexes"] = ",".join(str(x) for x in output_reuse_indexes)
        local_compile_options["ge.deterministic"] = "1" if torch.are_deterministic_algorithms_enabled() else "0"
        local_compile_options["ge.exec.atomicCleanPolicy"] = "1"
        logger.info("local compile options:")
        for k, v in local_compile_options.items():
            logger.info(f"  {k}: {v}")

        return local_compile_options, global_compile_options

    @staticmethod
    def _codegen_input(all_input_names, all_sym_input, input_func_list):
        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()
        all_input_str = ', '.join(all_input_names)
        if all_input_str:
            if len(all_input_names) == 1:
                all_input_str += ', '
            input_code.writeline(f'{all_input_str} = args')
        for name, idx in all_sym_input.items():
            if str(name).isdigit() or not isinstance(name, sympy.Symbol):
                # skip invalid expression, such as 2=arg0_1; s0*s1=arg1_1;
                continue
            input_code.writeline(f'{str(name)} = {all_input_names[idx]}')
        input_code.writeline(f'ge_inputs = [None] * {len(input_func_list)}')
        for idx, func in enumerate(input_func_list):
            if isinstance(func, _RngStatusInput):
                logger.info(f"skip codegen for rng input for ge input_{idx}.")
                continue
            input_code.writeline(f'ge_inputs[{idx}] = {func.codegen(all_input_names)}')
        return input_code.getvalue()

    @staticmethod
    def _gen_input_process(all_input_names, all_sym_input, input_func_list):
        from torch._inductor.utils import IndentedBuffer
        from types import ModuleType
        kernel = IndentedBuffer()
        kernel.writelines(['import torch', f'def kernel(*args):'])
        with kernel.indent():
            ge_inputs = GeConcreteGraph._codegen_input(all_input_names, all_sym_input, input_func_list)
            kernel.splice(ge_inputs)
            kernel.writeline('return ge_inputs')

        logger.info(f"input process func is: \n{kernel.getvalue()}")
        inputs_mod = ModuleType('inputs_mod')
        exec(compile(kernel.getvalue(), '<string>', 'exec'), inputs_mod.__dict__, inputs_mod.__dict__)
        normal_inputs_func = getattr(inputs_mod, 'kernel')

        unsupported_codegen_input = {}
        for idx, func in enumerate(input_func_list):
            if isinstance(func, _RngStatusInput):
                unsupported_codegen_input[idx] = func
        if len(unsupported_codegen_input) == 0:
            return normal_inputs_func

        def full_inputs_func(*args):
            ge_inputs = normal_inputs_func(*args)
            # update ge inputs for unsupported codegen input func
            for idx, func in unsupported_codegen_input.items():
                ge_inputs[idx] = func(args)
            return ge_inputs

        return full_inputs_func
