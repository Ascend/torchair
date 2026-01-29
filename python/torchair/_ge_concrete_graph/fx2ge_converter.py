from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Set
from datetime import datetime
import functools
import threading
import contextlib
from contextlib import contextmanager
import inspect
import sys
import time
import os
import warnings
import textwrap
import json
from pathlib import Path
import sympy
from packaging import version

import torch
from torch.fx.node import Argument, Target, Node
from torch import Tensor
from torch._C import NumberType
from torch._ops import OpOverload, OpOverloadPacket
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, is_fake

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable

from torchair.configs.compiler_config import CompilerConfig
from torchair.core import _torchair
from torchair.core._backend import initialize_graph_engine
from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger, EVENT_LEVEL
from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge import Clone
from torchair.ge.ge_custom import custom_op
from torchair.ge._ge_graph import ControlTensor, Tensor as GeTensor
from torchair.ge._ge_graph import get_default_ge_graph
from torchair.ge._ge_graph import _ValueInput, _TensorInput, _DiscontiguousTensorInput, _RngStatusInput, \
    _ValueType, _GeInputInfo, _SymPackInput
from torchair.ge._ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type, default_ge_graph, \
    GeGraph, attr_scope, compat_as_bytes, DataType, Format, TensorSpec, is_sym, sym_to_ge_dtype, assert_args_checkout
from torchair._ge_concrete_graph.graph_pass import optimize_sym_pack, optimize_reference_op_redundant_copy, \
    replace_data_to_refdata, get_frozen_flag, frozen_data_by_constplaceholder
from torchair._ge_concrete_graph.utils import convert_to_tensorboard, dump_graph, force_op_unknown_shape, \
    is_host_data_tensor, get_used_sym_value_mapping, Placement, compute_value_of_sym, \
    generate_sym_exper, get_sym_int_value, generate_shape_from_tensor, generate_real_shape_from_tensor,  \
    update_op_input_name_from_mapping, is_zero_element_tensor, flatten_meta_outputs, \
    make_real_tensor_like, get_used_syms_in_meta, _append_real_input_shape, is_complex
from torchair._ge_concrete_graph.hcom_utils import record_pg_to_graph, codegen_refresh_cache_pgname, \
    rename_cached_pgname
from torchair._ge_concrete_graph.supported_declaration import Support
from torchair._ge_concrete_graph.continguous_utils import guard_view_input
from torchair._ge_concrete_graph.export_config_generete import generate_config
from torchair._ge_concrete_graph.infer_symbol_shape import infer_and_gen_sym_shape_silent
from torchair._utils.export_utils import make_export_graph, get_export_file_name
from torchair._utils.graph_transform_observer import GraphTransformObserver
from torchair.inference._gear_utils import generate_dynamic_dims_option, get_dim_gears
from torchair.ge._ge_graph import compat_as_bytes, _ge_proto_dtype_to_ge_dtype, is_sym
from torchair.scope._scope_attr import guard_scope_attr
from . import ge_apis as ge


def _mapping_assign_op_to_graph_output(graph: GraphDef):
    """
    Maps Assign operations to graph outputs by updating input references.

    This function identifies Assign nodes in the graph and redirects their 
    outputs to existing graph inputs or existing Assign outputs.

    Args:
        graph (GraphDef): The graph definition to process.

    Returns:
        Dict[int, int]: Mapping from output reference indices to input indices.
    """    
    net_output: OpDef = None  # 输出节点
    net_inputs: Dict[str, int] = {}  # 输入tensor名称到索引的映射
    intput_ops: Dict[int, OpDef] = {}

    for op in graph.op:
        if op.type == "Data":
            net_inputs[GeTensor(op).tensor] = op.attr["index"].i
            intput_ops[op.attr["index"].i] = op
        elif op.type == "NetOutput":
            net_output = op
    if not net_output:
        raise AssertionError("NetOutput not found")

    def _mapping_to_graph_output(graph: GraphDef, graph_out: OpDef, assign_node_out: GeTensor,
                                 value_tensor: str, ref_intput_op: OpDef):
        raw_index_to_data_index = {}
        for i, name in enumerate(graph_out.input):
            if not name.endswith(":-1"):
                raw_index_to_data_index[i] = len(raw_index_to_data_index)
        output_ref_index_list = []
        for i, name in enumerate(graph_out.input):
            if name == assign_node_out.tensor:
                graph_out.input[i] = value_tensor
                output_ref_index_list.append(raw_index_to_data_index[i])
            elif name == value_tensor:
                output_ref_index_list.append(raw_index_to_data_index[i])
        if len(output_ref_index_list) != 0:
            return output_ref_index_list

        graph_out.input.append(value_tensor)
        graph_out.input_desc.add().CopyFrom(assign_node_out.desc)
        assign_out_shape = list(ref_intput_op.output_desc[0].shape.dim)
        graph_out.input_desc[-1].shape.dim[:] = assign_out_shape
        graph_out.input_desc[-1].name = f"input{len(graph_out.input_desc) - 1}"
        graph.attr["_output_dtypes"].list.i.append(_ge_proto_dtype_to_ge_dtype(assign_node_out.desc.dtype))
        return [len(graph_out.input_desc) - 1]

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
            ref_intput_op = intput_ops[net_inputs[op.input[0]]]
            output_ref_index_list = _mapping_to_graph_output(graph, net_output, assign_node_out,
                                                             op.input[1], ref_intput_op)
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
    Format.FORMAT_FRACTAL_Z_3D.value,
    Format.FORMAT_FRACTAL_NZ_C0_2.value,
    Format.FORMAT_FRACTAL_NZ_C0_4.value,
    Format.FORMAT_FRACTAL_NZ_C0_8.value
}
_DONT_EMPTY_TENSOR_OPT_OPS = {
    'npu_define.reduce_scatter_tensor_uneven.default',
    'npu_define.all_to_all_single_npu.default',
    'npu_define.allgather_in_tensor_uneven.default',
}

g_cached_dump_root: Optional[Dict[str, Any]] = None

DUMP_MAPPINGS = [
    ("ge.exec.dumpPath", "dump_path"),
    ("ge.exec.dumpMode", "dump_mode"),
    ("ge.exec.dumpData", "dump_data"),
    ("ge.exec.dumpLayer", "layer"),
    ("ge.exec.dumpStep", "dump_step"),
    ("ge.quant_dumpable", "quant_dumpable"),
]


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
    """
    Retrieves the converter function for a given PyTorch operation.

    Args:
        name (Callable): The PyTorch operation.

    Returns:
        Callable: The associated converter function or None if not found.
    """    
    if name not in _CONVERTERS:
        from torchair._ge_concrete_graph.ge_converter import custom
        if hasattr(name, "_ge_converter"):
            return name._ge_converter
    return _CONVERTERS[name]


def get_meta_outputs(meta_outputs):
    """
    Converts high-level tensor representations to tensor specifications.

    Args:
        meta_outputs: The meta outputs to convert.

    Returns:
        List[Any]: List of tensor specifications.
    """    
    if isinstance(meta_outputs, (list, tuple)):
        return [get_meta_outputs(meta_output) for meta_output in meta_outputs]
    return TensorSpec(meta_outputs)


def set_ge_outputs(ge_outputs, meta_outputs):
    """
    Sets metadata for GE outputs based on provided meta outputs.

    Args:
        ge_outputs: The GE output(s) to update.
        meta_outputs: The meta outputs providing metadata.
    """    
    if isinstance(ge_outputs, ge.Tensor):
        ge_outputs.set_meta(meta_outputs, ge_outputs)
    elif isinstance(ge_outputs, int):
        if not isinstance(meta_outputs, (torch.SymInt, int)):
            raise AssertionError("meta_outputs must be a torch.SymInt or an integer.")
    else:
        if not isinstance(ge_outputs, (list, tuple)):
            raise AssertionError("ge_outputs must be list or a tuple.")
        if not isinstance(meta_outputs, (list, tuple)):
            raise AssertionError("meta_outputs must be list or a tuple.")
        if len(ge_outputs) != len(meta_outputs):
            raise AssertionError("The length of ge_outputs must be equal to meta_outputs.")
        for meta_output, ge_output in zip(meta_outputs, ge_outputs):
            if meta_output is None:
                continue
            set_ge_outputs(ge_output, meta_output)


def _wrap_converter(converter: Callable):
    """
    Wraps a converter function to handle metadata processing.

    Args:
        converter (Callable): The converter function to wrap.

    Returns:
        Callable: The wrapped converter function.
    """    
    @functools.wraps(converter)
    def wrapped_converter(*args, **kwargs):
        meta_outputs = None
        if 'meta_outputs' in kwargs:
            meta_outputs = kwargs['meta_outputs']
            kwargs['meta_outputs'] = get_meta_outputs(meta_outputs)

        ge_outputs = converter(*args, **kwargs)

        return ge_outputs

    return wrapped_converter


class ExportSuccess(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f'ExportSucces: {self.message}'


class Converter:
    """
    Base class for converting PyTorch operations to GE operations.
    """    
    compile_backend = None
    result_checker = None

    def __init__(self, aten_op) -> None:
        self._aten_op = aten_op
        self._signature = inspect.signature(aten_op)
        self._supported_cases = None

    def __call__(self, converter) -> Any:
        """
        Registers a converter function for the associated PyTorch operation.

        Args:
            converter (Callable): The converter function to register.

        Returns:
            Any: The registered converter.
        """        
        wrapped_converter = _wrap_converter(converter)
        if 'meta_outputs' in inspect.signature(converter).parameters:
            wrapped_converter.require_meta = True
        else:
            wrapped_converter.require_meta = False
        try:
            self._aten_op._ge_converter = wrapped_converter
        except Exception:
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
                raise AssertionError("The testcase must be instance of Support")
        self._supported_cases = supported_cases


def declare_supported(supported_cases: List[Support]):
    def add_testcase(converter):
        if not isinstance(converter, Converter):
            raise AssertionError("The converter must be instance of Converter")
        converter.supported_cases = supported_cases
        _DECLARED_SUPPORTED_CONVERTERS.update({converter._aten_op: converter})
        return converter

    return add_testcase


def _declare_supported_converters():
    return _DECLARED_SUPPORTED_CONVERTERS


def empty_function(*args, **kwargs):
    pass


def register_fx_node_ge_converter(aten_op):
    """
    Registers a converter for a PyTorch operation.

    Args:
        aten_op: The PyTorch operation to register.

    Returns:
        Callable: Empty function if aten_op is None, else the Converter instance.
    """    
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
            if '_is_unfed_optional' not in desc.attr:
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
                if '_is_unfed_optional' not in desc.attr:
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

    for idx, _ in enumerate(runtime_inputs):
        if not (idx < len(input_index_mapping_graph_op)):
            raise AssertionError(
                f"GE graph input index {idx} out of Data ops index range {len(input_index_mapping_graph_op)}")

        if not runtime_inputs[idx].is_npu:
            logger.debug(f'input_{idx} is not npu tensor, skip format updates.')
            continue

        npu_format = torch_npu_module.get_npu_format(runtime_inputs[idx])
        origin_shape = list(runtime_inputs[idx].shape)
        if npu_format not in _SUPPORT_FORMAT_SET:
            raise RuntimeError(f"Unsupported input tensor with format {Format(npu_format).name}.")

        # attr "format_for_int" in proto::TensorDescriptor will be be deserialized as TensorDesc Format in ge.
        input_index_mapping_graph_op[idx].output_desc[0].attr["format_for_int"].i = npu_format
        input_index_mapping_graph_op[idx].input_desc[0].attr["format_for_int"].i = npu_format
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
        origin_format = None
        is_fractal_nz_c0 = npu_format in [Format.FORMAT_FRACTAL_NZ_C0_2.value,
                                           Format.FORMAT_FRACTAL_NZ_C0_4.value,
                                           Format.FORMAT_FRACTAL_NZ_C0_8.value]
        if npu_format == Format.FORMAT_FRACTAL_NZ.value or is_fractal_nz_c0:
            origin_format = Format.FORMAT_ND.value
        elif npu_format == Format.FORMAT_FRACTAL_Z.value or npu_format == Format.FORMAT_NC1HWC0.value:
            origin_format = Format.FORMAT_NCHW.value
        elif npu_format == Format.FORMAT_FRACTAL_Z_3D.value or npu_format == Format.FORMAT_NDC1HWC0.value:
            origin_format = Format.FORMAT_NCDHW.value

        if origin_format is not None:
            input_index_mapping_graph_op[idx].output_desc[0].attr["origin_format_for_int"].i = origin_format

            from torch_npu.npu.utils import _is_gte_cann_version
            if _is_gte_cann_version("8.2.RC1"):
                input_index_mapping_graph_op[idx].attr["_enable_storage_format_spread"].b = False
                input_index_mapping_graph_op[idx].output_desc[0].shape.dim[:] = []
                input_index_mapping_graph_op[idx].output_desc[0].attr["origin_shape"].list.val_type = 2
                input_index_mapping_graph_op[idx].output_desc[0].attr["origin_shape"].list.i.extend(origin_shape)
                input_index_mapping_graph_op[idx].output_desc[0].attr["origin_shape_initialized"].b = True
                input_index_mapping_graph_op[idx].output_desc[0].attr["origin_format_is_set"].b = True

                input_index_mapping_graph_op[idx].input_desc.pop(0)
                input_index_mapping_graph_op[idx].input_desc.add().CopyFrom(
                    input_index_mapping_graph_op[idx].output_desc[0])
                input_index_mapping_graph_op[idx].input_desc[0].name = 'x'

        logger.debug(f'update the Format of output TensorDesc for input_{idx} to Format {Format(npu_format).name}.')


def _update_constplaceholder_attr_from_inputs(graph: GraphDef, runtime_inputs):
    update_node_name_mapping = {}
    for op in graph.op:
        if op.type == "ConstPlaceHolder":
            if 'update_node_from_fx_input_idx' not in op.attr:
                logger.debug(f'No need to update {op.name}.')
                continue
            real_input = runtime_inputs[op.attr["update_node_from_fx_input_idx"].i]
            if not real_input.is_npu:
                raise AssertionError("ConstPlaceHolder only support npu tensor.")
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
    def __init__(self, index, meta_output, sym_value_mapping, is_view_as_real, is_view_as_complex):
        self._fx_input_index = index
        self._sym_value_mapping = sym_value_mapping
        self._ori_meta_shape = list(meta_output.size())
        self._ori_meta_stride = list(meta_output.stride())
        self._ori_meta_offset = meta_output.storage_offset()
        self._meta_output_shape = generate_sym_exper(list(meta_output.size()))
        self._meta_output_stride = generate_sym_exper(list(meta_output.stride()))
        self._meta_output_offset = generate_sym_exper(meta_output.storage_offset())
        self._is_view_as_real = is_view_as_real
        self._is_view_as_complex = is_view_as_complex

    @property
    def is_view_as_real(self):
        return self._is_view_as_real

    @property
    def is_view_as_complex(self):
        return self._is_view_as_complex

    def compute_output(self, *args):
        real_input = args[self._fx_input_index]
        if self._is_view_as_real:
            return torch.view_as_real(real_input)
        if self._is_view_as_complex:
            return torch.view_as_complex(real_input)
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


def get_or_auto_gen_converter(target):
    if hasattr(target, "_ge_converter"):
        converter = target._ge_converter
    else:
        converter = _get_converter(target)
    if converter is not None:
        return converter    
    
    if _can_autogenerate_converter(target):
        converter_code = _generate_converter_code(target)
        logger.info(f"Ascend op converter has auto generated: {converter_code}")
        exec(converter_code)
        converter = target._ge_converter
        return converter
    else:
        raise RuntimeError(f"Ascend op converter is not implemented of: {target}")
        

def _can_autogenerate_converter(target):
    if _is_torch_custom(target):
        converter_code = _generate_converter_log
        if _has_no_scalar(target):
            target_name = str(target).split(".")[1]
            if target_name.split("_")[-1] == "functional":
                target_name.pop()
            ge_name = "".join(word.capitalize() for word in target_name.split("_"))
            (status, _, _, _) = _torchair.get_registered_ir_def(ge_name)
            if status == "None":
                return False
            if status != "SUCCESS":
                raise RuntimeError(f"Failed to converter {target} to AscendIR: can not find registered "
                                   f"AscendIR {ge_name}, its need to meet the upper camel case format, "
                                   f"please implement this function and ensure AscendIR "
                                   f"has been registered correctly.{converter_code}")
            return True
        else:
            raise RuntimeError(f"Failed to converter {target} to AscendIR: this op has scalar input, "
                               f"can not auto generate converter, "
                               f"please implement this function.{converter_code}")
    return False


def _is_torch_custom(target):
    if isinstance(target, OpOverload):
        if any(s in str(target) for s in ["prim", "prims", "aten"]) or is_builtin_callable(target):
            raise RuntimeError(
                f"Failed to converter {target} to AscendIR: this op is not custom op to auto generate, "
                f"need to be implemented in oringinal converter register or implement new converter register.")
        else:
            return True
    else:
        return False
    

def _generate_converter_log(target):
    target_log_name = str(target).split(".")[1]
    if target_log_name.split("_")[-1] == "functional":
        target_log_name = "_".join(target_log_name.split("_")[:-1])
    target_args_params, target_kwargs_name, _ = _get_target_params(target)
    if len(target_kwargs_name) > 0:
        params_code = ',\n    '.join(target_args_params) + ",\n    *,\n    " + ',\n    '.join(target_kwargs_name)
    else:
        params_code = ',\n    '.join(target_args_params)
    function_log = textwrap.dedent('''
        @register_fx_node_ge_converter({op_name})
        def {func_name}(
            {params}
        ):
            #inplement AscendIR converter here.
        ''').format(
        op_name='torch.ops.' + str(target),
        func_name='converter' + '_' + target_log_name,
        params=params_code
    )
    return function_log


def _has_no_scalar(target):
    for arg in target._schema.arguments:
        if isinstance(arg.type, NumberType):
            return False
    return True


def _alias_is_write(target):
    alias_list = []
    for i, arg in enumerate(target._schema.arguments):
        if arg.alias_info is not None:
            if arg.alias_info.is_write:
                alias_list.append(i)
    return alias_list


def _ge_inplace(inputs, outputs):
    inplace_outputs_index = []
    inplace_inputs_index = []
    ge_inputs_dict, ge_outputs_dict = dict(inputs), dict(outputs)
    for index, key in enumerate(ge_outputs_dict.keys()):
        if key in ge_inputs_dict.keys():
            inplace_outputs_index.append(index)
    for index, key in enumerate(ge_inputs_dict.keys()):
        if key in ge_outputs_dict.keys():
            inplace_inputs_index.append(index)
    return inplace_outputs_index, inplace_inputs_index


def _get_target_params(target):
    #构造args的参数列表和kwargs的参数列表
    target_args_params = []
    target_kwargs_name = []
    #根据tensor和其它类型分类排序
    args_tensor_name = []
    args_notensor_name = []
    for arg in target._schema.arguments:
        if not arg.kwarg_only:
            if arg.has_default_value():
                target_args_params.append(arg.name + "=" + repr(arg.default_value))
            else:
                target_args_params.append(arg.name)
        else:
            if arg.has_default_value():
                target_kwargs_name.append(arg.name + "=" + repr(arg.default_value))
            else:
                target_kwargs_name.append(arg.name)
        if str(arg.type) == "Tensor" or str(arg.type) == "Optional[Tensor]":
            args_tensor_name.append(arg.name)
        else:
            args_notensor_name.append(arg.name)
    args_all_name = args_tensor_name + args_notensor_name
    return target_args_params, target_kwargs_name, args_all_name


def _generate_converter_code(target):
    target_name = str(target).split(".")[1]
    if target_name.split("_")[-1] == "functional":
        target_name = "_".join(target_name.split("_")[:-1])
    ge_name = "".join(word.capitalize() for word in target_name.split("_"))
    (_, ge_inputs, ge_outputs, _) = _torchair.get_registered_ir_def(ge_name)

    #处理参数
    target_args_params, target_kwargs_name, args_all_name = _get_target_params(target)
    if len(target_kwargs_name) > 0:
        params_code = ',\n    '.join(target_args_params) + ",\n    *,\n    " + ',\n    '.join(target_kwargs_name)
    else:
        params_code = ',\n    '.join(target_args_params)

    #处理inpalce
    need_clone = False
    need_reduce_output = False
    alias_list = _alias_is_write(target) # target上的inplace的位置
    inplace_outputs_index, inplace_inputs_index = _ge_inplace(ge_inputs, ge_outputs) # ge上的inplace的位置
    real_output = [i for i in range(len(ge_outputs))] # 所有输出的位置
    if len(alias_list) != 0 and len(alias_list) != len(inplace_outputs_index):
        raise RuntimeError(
            f"Failed to converter {target} to AscendIR: the number of inplace inputs for torch does not "
            f"match the AscendIR {ge_name}, please check your torch and AscendIR registration.")
    #如果target未定义成inpalce的输入，则处理ge中需要做clone的参数
    if len(alias_list) == 0 and len(inplace_outputs_index) != 0:
        need_clone = True
        clone_code = []
        for index in inplace_inputs_index:
            clone_code.append(str(args_all_name[index]) + '= Clone(' + str(args_all_name[index]) + ')')
    #如果target定义成inpalce的输入，则走functional的逻辑，删除inplace的输出只返回非inplace
    if len(alias_list) != 0 and len(alias_list) == len(inplace_outputs_index):
        need_reduce_output = True
        for index in inplace_outputs_index:
            real_output.remove(index)
        if len(real_output) == 0 :
            reduce_code = ""
        else:
            reduce_code = "return (" + ", ".join([f"out[{n}]" for n in real_output]) + ")"

    imports = textwrap.dedent('''
    # Auto-generated from {target}, not edit
    from torchair._ge_concrete_graph.ge_converter.converter_utils import *
    ''').format(target=str(target))
    if need_reduce_output:
        function = textwrap.dedent('''
        @register_fx_node_ge_converter({op_name})
        def {func_name}(
            {params}
        ):
            out = custom_op(
                {ascendir},{ascend_params}
                )
            {output}
        ''').format(
            op_name='torch.ops.' + str(target),
            func_name='converter' + '_' + target_name,
            params=params_code,
            ascendir='"' + ''.join(word.capitalize() for word in target_name.split('_')) + '"',
            ascend_params=", ".join(args_all_name),
            output=reduce_code
        )
    elif need_clone:
        function = textwrap.dedent('''
        @register_fx_node_ge_converter({op_name})
        def {func_name}(
            {params}
        ):
            {clone_arg}
            return custom_op(
                {ascendir},{ascend_params}
                )
        ''').format(
            op_name='torch.ops.' + str(target),
            func_name='converter' + '_' + target_name,
            params=params_code,
            clone_arg='\n'.join(clone_code),
            ascendir='"' + ''.join(word.capitalize() for word in target_name.split('_')) + '"',
            ascend_params=", ".join(args_all_name)
        )
    else:
        function = textwrap.dedent('''
        @register_fx_node_ge_converter({op_name})
        def {func_name}(
            {params}
        ):
            return custom_op(
                {ascendir},{ascend_params}
                )
        ''').format(
            op_name='torch.ops.' + str(target),
            func_name='converter' + '_' + target_name,
            params=params_code,
            ascendir='"' + ''.join(word.capitalize() for word in target_name.split('_')) + '"',
            ascend_params=", ".join(args_all_name)
        )

    code = f"{imports}\n{function}"
    return code


class GeConcreteGraph(ConcreteGraphBase):
    """
    GeConcreteGraph represents a concrete computation graph optimized for Ascend NPU devices.
    It extends the base ConcreteGraphBase to provide GE-specific compilation and execution capabilities.
    
    Args:
        config (CompilerConfig): Configuration object for compiler settings.
        name (str, optional): Name of the graph. Defaults to "graph".
    """       
    def __init__(self, config: CompilerConfig, name=None):
        self._graph = GeGraph(name=name)
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._fx_input_names = []
        self._input_process = None
        self._input_info_list = []
        self._graph_output_ref_input = {}
        self._ref_data_idx = []
        self._cloned_ge_input_mapping = {}
        self._config = config
        self._auto_tune_times = 0
        self._converter_ctx = threading.local()
        self._is_compiled = False
        self._all_sym_input_idx = {}
        self._all_sym_input_data = {}
        self._all_meta_tensor_input = {}
        self._fx_graph = None
        self._has_empty_tensor = False
        self._scope_attr_ctx = threading.local()        
        self._muti_stream_op_order = defaultdict(list)
        self._wait_control_edges = defaultdict(list)         
        _, global_compile_options = self._normalize_ge_option()
        initialize_graph_engine(global_compile_options)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the graph with the provided inputs.
        
        This method handles input processing, graph execution, and output retrieval.
        It supports dynamic shape handling, reference data management, and automatic tuning.
        
        Args:
            *args: Variable length argument list for graph inputs.
            **kwargs: Arbitrary keyword arguments for graph inputs.
        
        Returns:
            Any: Output tensors from the executed graph.
        
        Raises:
            ExportSuccess: If graph export is triggered during execution.
        """        
        enable_event_log = logger.getEffectiveLevel() <= EVENT_LEVEL
        t_begin = time.time() if enable_event_log else 0
        if not self._is_compiled:
            # Equivalent functionality to 'self._input_func_list[i](*args)', but better performance.
            _input_func_list = [info.func for info in self._input_info_list]
            self._input_process = self._gen_input_process(self._fx_input_names, self._all_sym_input_idx,
                                                          _input_func_list)
        inputs = self._input_process(*args)
        t_input_process = time.time() if enable_event_log else 0

        self.update_graph_with_runtime(inputs, args)

        if self.config.export.export_mode:
            self.export(inputs)
            raise ExportSuccess("export graph over")

        if not self._is_compiled:
            local_compile_options, global_compile_options = self._normalize_ge_option()
            ge_data_dump_options = _get_ge_dump_options(global_compile_options) # _get_ge_dump_options must execute before initialize_graph_engine
            initialize_graph_engine(global_compile_options)
            _set_dump_options_json(ge_data_dump_options, self._graph)
            self.graph.load(local_compile_options)

        if self.should_auto_tune:
            self.auto_tune(inputs)

        self.compile()
        t_compile = time.time() if enable_event_log else 0
        if len(self._graph_output_ref_input):
            assigned_outputs = [None] * len(self.graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in self._graph_output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]
            ge_outputs = self.graph.run(inputs, assigned_outputs)
        else:
            ge_outputs = self.graph.run(inputs)
        t_run = time.time() if enable_event_log else 0
        logger.event("torchair run at %s, process input: %sus, compile %sus, graph run: %sus",
                     int(t_begin * 1e6),
                     int((t_input_process - t_begin) * 1e6),
                     int((t_compile - t_input_process) * 1e6),
                     int((t_run - t_compile) * 1e6))

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

    def optimize_graph_without_runtime(self, *sample_args, observer=None, aot_gm=None):
        """
        Optimizes the computation graph without relying on runtime information.
        This includes passes like dead data removal, explicit ordering for side-effect nodes,
        and reference data optimization.
        
        Args:
            *sample_args: Sample input arguments for tracing and optimization.
        """        
        from torchair._ge_concrete_graph.graph_pass import remove_dead_data_and_reorder_data_index
        from torchair._ge_concrete_graph.graph_pass import explicit_order_for_side_effect_nodes
        from torchair._ge_concrete_graph.graph_pass import explicit_order_for_cmo
        from torchair._ge_concrete_graph.utils import get_graph_input_placements
        observer.dump_gegraph(self.graph, "original_ge_graph")

        observer.apply_gegraph_pass(optimize_sym_pack, self.graph, "optimize_sym_pack")
        record_pg_to_graph(self.graph)

        # Note:
        # Please do not take any actions to add or delete data nodes, or change the index of data nodes after this.
        self._input_info_list = observer.apply_gegraph_pass(
            remove_dead_data_and_reorder_data_index, self.graph, "remove_dead_data_and_reorder_data_index")

        self.graph.attr["_input_placements"].list.i.extend(get_graph_input_placements(self.graph))
        self.graph.attr["_output_dtypes"].list.i.extend([output.dtype for output in self.outputs])
        self.graph.attr["_executor_type"].i = _get_executor_type()

        self._ref_data_idx = observer.apply_gegraph_pass(
            optimize_reference_op_redundant_copy, self.graph, "optimize_reference_op_redundant_copy")
        self._graph_output_ref_input = observer.apply_gegraph_pass(
            _mapping_assign_op_to_graph_output, self.graph, "mapping_assign_op_to_graph_output")

        for ge_index, input_info in enumerate(self._input_info_list):
            if isinstance(input_info.func, _DiscontiguousTensorInput):
                self._cloned_ge_input_mapping[ge_index] = input_info.func.fx_input_idx

        # Note: The following two passes must be executed after the above pass.
        explicit_order_for_side_effect_nodes_partial = functools.partial(
            explicit_order_for_side_effect_nodes, graph_output_ref_input=self._graph_output_ref_input)
        observer.apply_gegraph_pass(
            explicit_order_for_side_effect_nodes_partial, self.graph, "explicit_order_for_side_effect_nodes")
        observer.apply_gegraph_pass(
            explicit_order_for_cmo, self.graph, "explicit_order_for_cmo")
        
        _normalize_ge_graph(self.graph)
        observer.dump_gegraph(self.graph, "optimized_ge_graph")

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
            for _, v in self._graph_output_ref_input.items():
                ref_data_idx.add(v)
            replace_data_to_refdata(self.graph, ref_data_idx, inputs)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path(f"dynamo_optimized_{self.graph.name}"))

    def compile(self) -> Any:
        """
        Compiles the computation graph into an executable GE graph.

        This method finalizes the graph structure and prepares it for execution.
        It ensures all nodes are properly configured and dependencies are resolved.
        """
        if self._is_compiled:
            return

        logger.info(f'start compile graph: {self.graph.name}.')
        self.graph.compile()
        self._is_compiled = True
        logger.info(f'end compile graph: {self.graph.name} and start run graph.')

    def codegen(self, extend_config, enable_cache=False):
        """
        Generates executable code from the optimized graph.
        
        This method is used for code generation in JIT compilation scenarios.
        It supports caching mechanisms to improve subsequent compilation speed.
        
        Args:
            extend_config (Dict[str, Any]): Additional configuration for code generation.
            enable_cache (bool, optional): Flag to enable caching. Defaults to False.
        
        Returns:
            Optional[str]: Generated code as a string if successful, otherwise None.
        """        
        from torch._inductor.utils import IndentedBuffer
        if self._config.experimental_config.enable_ref_data or self.config.export.export_mode \
                or self.config.aoe_config.aoe_mode.value is not None:
            logger.info(f'Unsupported codegen for graph with config enable_ref_data or export_mode or aoe_mode.')
            return None

        head = IndentedBuffer()
        head.splice('''
        import torch
        import os
        import numpy
        from torchair.core._backend import initialize_graph_engine
        from torchair.ge._ge_graph import GeGraph
        from torchair._ge_concrete_graph.fx2ge_converter import _update_constplaceholder_attr_from_inputs
        from torchair._ge_concrete_graph.fx2ge_converter import _update_internal_format_from_inputs
        assert_size_stride = torch._C._dynamo.guards.assert_size_stride
        ''')
        need_rebuild_pg = enable_cache and (len(self.graph.used_process_group) != 0)
        if need_rebuild_pg:
            if extend_config.get("ge.graph_compiler_cache_dir") is not None:
                head.splice(f'''
                import torch_npu
                if not torch_npu.distributed._is_support_hccl_comm_name():
                    raise AssertionError(f"During cache loading, "
                                         f"it is not possible to create a PG with the same name in the ge cache. "
                                         f"This may be due to the low version of CANN you are using. "
                                         f"Please upgrade and try again.")
                    ''')
            cache_graph = GeGraph()
            cache_graph.MergeFrom(self.graph._proto)
            rename_cached_pgname(cache_graph._proto, self.graph.used_process_group)
            head.writelines(['', f'serialized_graph = {cache_graph.SerializeToString()}'])
        else:
            head.writelines(['', f'serialized_graph = {self.graph.SerializeToString()}'])
        head.writelines(['', f'global_compile_options = {{}}'])
        local_compile_options, global_compile_options = self._normalize_ge_option()
        for k, v in global_compile_options.items():
            head.writeline(f'global_compile_options["{k}"] = "{v}"')
        head.writeline(f'local_compile_options = {{}}')
        for k, v in local_compile_options.items():
            head.writeline(f'local_compile_options["{k}"] = "{v}"')
        for k, v in extend_config.items():
            head.writeline(f'local_compile_options["{k}"] = "{v}"')
        if extend_config.get("ge.graph_compiler_cache_dir") is not None:
            head.writeline(f'local_compile_options["ge.graph_key"] = "{self._graph.name}"')
            head.splice(f'''
            def _update_ge_cache_dir(path):
                local_compile_options["ge.graph_compiler_cache_dir"] = path
            ''')
        head.writelines(['', 'initialize_graph_engine(global_compile_options)',
                         'ge_graph = GeGraph(serialized_model_def=serialized_graph)'])
        if need_rebuild_pg:
            create_pgname_code = codegen_refresh_cache_pgname(self.graph.used_process_group)
            head.splice(create_pgname_code)
        kernel = IndentedBuffer()
        kernel.writelines(['', '_is_first_run = True', f'def kernel(*args):'])
        with kernel.indent():
            _input_func_list = [info.func for info in self._input_info_list]
            input_code = GeConcreteGraph._codegen_input(self._fx_input_names, self._all_sym_input_idx, _input_func_list,
                                                        enable_cache)
            kernel.splice(input_code)

            # for info update in first run
            kernel.writelines(['', 'global _is_first_run', 'if _is_first_run:'])
            with kernel.indent():
                kernel.writelines(['_is_first_run = False'])
                assert_code = self._codegen_assert_size_stride(self._all_meta_tensor_input)
                kernel.splice(assert_code)
                kernel.writelines(['_update_constplaceholder_attr_from_inputs(ge_graph, args)',
                                   '_update_internal_format_from_inputs(ge_graph, ge_inputs)',
                                   'ge_graph.load(local_compile_options, create_pg=False)',
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
                        kernel.writeline(f'args[{fx_index}].copy_(ge_inputs[{ge_index}])')

            kernel.writeline(f'fx_outputs = [None] * {len(self._fx_outputs)}')
            for idx, out in enumerate(self._fx_outputs):
                if isinstance(out, ViewOfInput):
                    if out.is_view_as_real:
                        kernel.writeline(
                            f'fx_outputs[{idx}] = torch.view_as_real(args[{out._fx_input_index}])')
                    elif out.is_view_as_complex:
                        kernel.writeline(
                            f'fx_outputs[{idx}] = torch.view_as_complex(args[{out._fx_input_index}])')
                    else:
                        kernel.writeline(
                            f'fx_outputs[{idx}] = torch.as_strided(args[{out._fx_input_index}], {out._ori_meta_shape}, '
                            f'{out._ori_meta_stride}, {out._ori_meta_offset})')
                elif isinstance(out, SymOutput):
                    kernel.writeline(f'fx_outputs[{idx}] = {str(out._ori_meta_sym.node.expr)}')
                elif not isinstance(out, GeTensor):
                    kernel.writeline(f'fx_outputs[{idx}] = {self._fx_outputs[idx]}')
                else:
                    if idx not in self._fx_outputs_mapping.keys():
                        raise AssertionError("The index of _fx_outputs is not in _fx_outputs_mapping.keys()")
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
        """
        Parses input metadata during graph construction.
        
        This method processes input tensors and their associated metadata to configure graph inputs.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
        
        Returns:
            Any: Processed input data for the graph.
        """        
        data_index = self.graph.num_inputs
        self._fx_input_names.append(target)
        if is_sym(meta_outputs):
            self._all_sym_input_idx[(meta_outputs).node.expr] = data_index
            data = ge.Data(index=data_index, dtype=sym_to_ge_dtype(meta_outputs), shape=[], placement='CPU',
                           node_name=target)
            data.set_meta(meta_outputs)
            self._all_sym_input_data[str((meta_outputs).node.expr)] = data
            input_info = _GeInputInfo(value_type=_ValueType.TENSOR, func=_ValueInput(data_index), shape=[],
                                      device_type="CPU", real_shape=[])
        else:
            if not isinstance(meta_outputs, torch.Tensor):
                raise AssertionError("meta_outputs must be instance of torch.Tensor")
            self._all_meta_tensor_input[data_index] = meta_outputs
            dtype = torch_type_to_ge_type(meta_outputs.dtype)
            shape = generate_shape_from_tensor(meta_outputs)
            real_shape = generate_real_shape_from_tensor(meta_outputs)
            placement = 'CPU' if (meta_outputs.device is None or meta_outputs.device.type == 'cpu') else 'NPU'
            data = ge.Data(index=data_index, dtype=dtype, shape=shape, placement=placement, node_name=target)
            data.set_meta(meta_outputs)
            value_type = _ValueType.TENSOR
            if isinstance(meta_outputs, torch.nn.Parameter) or hasattr(meta_outputs, "_torchair_is_parameter"):
                value_type = _ValueType.PARAMETER
            elif getattr(meta_outputs, "_dynamo_static_input_type", None) == "guarded":
                if any(is_sym(shape) for shape in meta_outputs.size()):
                    raise AssertionError(
                        f"The mark_static_address input tensor need be static shape, "
                        f"input: {target}, size: {meta_outputs.shape}")
                value_type = _ValueType.STATIC_TENSOR

            input_info = _GeInputInfo(
                value_type=value_type,
                func=_TensorInput(data_index) if meta_outputs.is_contiguous() else _DiscontiguousTensorInput(
                    data_index),
                shape=shape, dim_gears=get_dim_gears(meta_outputs) or {}, device_type=placement,
                real_shape=real_shape)
       
        self.graph.record_input_info(data.node.name, input_info)
        return data

    @guard_view_input
    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        """
        Parses output metadata during graph construction.
        
        This method processes output tensors and their associated metadata to configure graph outputs.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
        
        Returns:
            Any: Processed output data for the graph.
        """        
        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise AssertionError("The input args must be list or a tuple, and the length of args must be euqal to 1")
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
                    is_view_as_complex = (is_complex(arg.meta) and not is_complex(fx_input_meta))
                    is_view_as_real = (not is_complex(arg.meta) and is_complex(fx_input_meta))
                    self._fx_outputs[-1] = ViewOfInput(fx_input_idx, arg.meta,
                                                       get_used_sym_value_mapping(self._all_sym_input_idx, arg.meta),
                                                       is_view_as_real, is_view_as_complex)
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

        ge.NetOutput(self._outputs, dependencies=self.graph.must_keep_ops)
        return args

    def parse_symlist(self, syms):
        """
        Parses a list of symbols (either integers or ValuePack objects) into a list of NPU-compatible symbols.
        
        Args:
            syms (List[Union[int, ValuePack]]): List containing symbols to parse.
        
        Returns:
            List[int]: Parsed list of integer symbols.
        """        
        npu_syms = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                npu_syms.append(sym.npu)
            else:
                if not isinstance(sym, int):
                    raise AssertionError("sym must be an integer.")
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
    @guard_scope_attr
    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        """
        Parses individual nodes within the graph during compilation.
        
        This method handles custom operation conversion and optimization passes.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
        
        Returns:
            Any: Processed result of the parsed node.
        """
        if str(target) in ['air.scope_enter.default', 'air.scope_exit.default']:
            return target(*args, **kwargs, need_execute=True)
        if str(target) == 'air.record.default':
            return self._build_record_control_edge()
        if str(target) == 'air.wait.default':
            self._build_wait_control_edge(args)
            return None
                
        all_zero_and_nosym = all([is_zero_element_tensor(t) and not get_used_syms_in_meta(t)
            for t in flatten_meta_outputs(meta_outputs)])
        if all_zero_and_nosym and (str(target) not in _DONT_EMPTY_TENSOR_OPT_OPS):
            return make_real_tensor_like(meta_outputs)

        if not self._has_empty_tensor:
            self._has_empty_tensor = all([is_zero_element_tensor(t) for t in flatten_meta_outputs(meta_outputs)])

        graph = get_default_ge_graph()
        num_ops = len(graph.op)

        if str(target) in ('aten.sym_stride.int', 'aten.sym_storage_offset.default'):
            from .infer_symbol_calculate import infer_ge_output_by_symbol_calculate
            ge_outputs = infer_ge_output_by_symbol_calculate(self._all_sym_input_data, meta_outputs)
        elif str(target) == 'auto_functionalized_v2':
            from .auto_functionalized_v2 import conveter_auto_functionalize_v2
            kwargs['symbol_input_map'] = self._all_sym_input_data
            ge_outputs = conveter_auto_functionalize_v2(*args, **kwargs)
        else:
            converter = get_or_auto_gen_converter(target)
            if converter.require_meta:
                kwargs['meta_outputs'] = meta_outputs
            ge_outputs = converter(*args, **kwargs)
            infer_and_gen_sym_shape_silent(target, args, kwargs, ge_outputs, graph.op[num_ops:])        

        self._handle_wait_control_edge(graph.op[num_ops:])        
        if meta_outputs is not None:
            set_ge_outputs(ge_outputs, meta_outputs)
            if hasattr(self._converter_ctx, 'node') and self._converter_ctx.node:
                fx_tensor_prefix = f'{self._converter_ctx.node.name}-{self._converter_ctx.node.target}.OUTPUT'
                if isinstance(ge_outputs, ge.Tensor):
                    ge_outputs.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.0')
                elif isinstance(ge_outputs, (list, tuple)) and all([isinstance(v, ge.Tensor) for v in ge_outputs]):
                    for i, ge_output in enumerate(ge_outputs):
                        ge_output.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.{i}')
        return ge_outputs

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
        if self.config.export.experimental.auto_atc_config_generated:
            generate_config(self.config.export.export_name, file_path, export_graph.used_process_group)
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

    @property
    def fx_graph(self):
        return self._fx_graph

    def save_fx_graph(self, graph_module: torch.fx.GraphModule):
        self._fx_graph = graph_module

    @contextmanager
    def converter_context(self, *, node):
        def stringify_shape(shape):
            return f"[{', '.join([str(x) for x in shape])}]"

        def format_node(node):
            from torch.fx.experimental.proxy_tensor import py_sym_types
            from torch.fx.passes.shape_prop import TensorMetadata
            if not isinstance(node, Node):
                return node
            meta_val = node.meta.get("val", node.meta.get("tensor_meta", node.meta.get("example_value", None)))
            # use string as annotation, to make it valid python code
            if isinstance(meta_val, torch.Tensor):
                node_dtype = meta_val.dtype
                node_shape = stringify_shape(meta_val.shape)
                node_device = meta_val.device
                return f'{node.name}: {node_dtype}{node_shape}{node_device}'
            elif isinstance(meta_val, py_sym_types):
                return f'{node.name}: "Sym({meta_val})"'
            elif isinstance(meta_val, TensorMetadata):
                node_dtype = meta_val.dtype
                node_shape = stringify_shape(meta_val.shape)
                return f'{node.name}: {node_dtype}{node_shape}'
            else:
                return node.name           

        if node.stack_trace is not None:
            file_line = node.stack_trace.split(' File ')[-1].replace('\n', '')
            if file_line not in self.graph._python_code:
                self.graph._python_code += f'\n# File {file_line}\n'
            node_target = node._pretty_print_target(node.target)
            formatted_node = format_node(node)
            format_args = []
            format_kwargs = []
            for node_arg in node.args:
                format_args.append(format_node(node_arg))
            for kwarg_k, kwarg_v in node.kwargs.items():
                format_args.append(f'{kwarg_k}={format_node(kwarg_v)}')
            self.graph._python_code += \
                f'## FX Code: ' \
                f'{self.graph.format_python_code(formatted_node, node_target, None, format_args, format_kwargs)}\n'

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

    @contextmanager
    def scope_attr_ctx(self, scope_attr_info):        
        try:
            self._scope_attr_ctx.scope_attr = scope_attr_info
            yield
        finally:
            self._scope_attr_ctx.scope_attr = None              

    def auto_tune(self, inputs: List[Tensor]) -> Any:
        logger.info(f"Start auto tune for round {self._auto_tune_times}")
        self.graph.auto_tune(inputs)
        self._auto_tune_times += 1
        logger.info(f"End auto tune for round {self._auto_tune_times - 1}")

    def _normalize_ge_option(self):
        # Initialize based on global options
        local_compile_options, global_compile_options = self.config.as_dict()
        global_compile_options["ge.exec.staticMemoryPolicy"] = "2"
        global_compile_options["ge.opSelectImplmode"] = "high_precision"
        logger.info("global compile options:")
        for k, v in global_compile_options.items():
            logger.info(f"  {k}: {v}")

        # Update local options
        output_reuse_indexes = [x for x in range(len(self.outputs)) if x not in self._graph_output_ref_input.keys()]
        if len(output_reuse_indexes) != 0:
            # support output memory reuse while output is not ref to input
            local_compile_options["ge.exec.outputReuseMemIndexes"] = ",".join(str(x) for x in output_reuse_indexes)
        if self._graph_output_ref_input:
            # supports memory reuse options between input and output of the muted inplace operator
            local_compile_options["ge.exec.outputReuseInputMemIndexes"] = (
                "|".join(f"{input_},{output_}" for output_, input_ in self._graph_output_ref_input.items())
            )
        local_compile_options["ge.deterministic"] = "1" if torch.are_deterministic_algorithms_enabled() else "0"
        local_compile_options["ge.exec.atomicCleanPolicy"] = "1"
        local_compile_options.update(generate_dynamic_dims_option(self.graph.named_inputs_info,
                                     self.config.inference_config.dynamic_gears_merge_policy.value))
        if self._config.experimental_config.frozen_parameter:
            if version.parse(torch.__version__) < version.parse("2.5.1"):
                warnings.warn('When enable frozen_parameter, Parameters will be considered frozen. '
                              'Please make sure that the Parameters data address remain the same '
                              'throughout the program runtime.')
            else:
                warnings.warn('When enable frozen_parameter, Parameters and input tensors with immutable data_ptr '
                              'marked by `torch._dynamo.mark_static_address()` will be considered frozen. '
                              'Please make sure that the Parameters data address remain the same '
                              'throughout the program runtime.')
            optimize_frozen_flag_list = get_frozen_flag(self._input_info_list)
        else:
            optimize_frozen_flag_list = [0] * len(self._input_info_list)
        if len(optimize_frozen_flag_list) != 0:
            local_compile_options["frozenInput"] = ",".join(str(x) for x in optimize_frozen_flag_list)
        local_compile_options["ge.exec.allTensorNotEmpty"] = '0' if self._has_empty_tensor else '1'
        local_compile_options.update(_append_real_input_shape(self._input_info_list))
        logger.info("local compile options:")
        for k, v in local_compile_options.items():
            logger.info(f"  {k}: {v}")

        return local_compile_options, global_compile_options

    @staticmethod
    def _codegen_input(all_input_names, all_sym_input, input_func_list, enable_cache=False):
        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()

        if all_sym_input:
            # only for cache
            all_input_str = ', '.join(all_input_names)
            if all_input_str:
                if len(all_input_names) == 1:
                    all_input_str += ', '
            input_code.writeline(f'{all_input_str} = args')
            for name, idx in all_sym_input.items():
                if str(name).isdigit() or not isinstance(name, sympy.Symbol):
                    continue
                input_code.writeline(f'{str(name)} = {all_input_names[idx]}')

        input_code.writeline('ge_inputs = list(args)')
        func_idx = []
        for _, func in enumerate(input_func_list):
            fx_input_idx = getattr(func, 'fx_input_idx', None)
            if fx_input_idx is not None:
                func_idx.append(fx_input_idx)
        indices_to_remove = []
        for i in range(len(all_input_names)):
            if i not in func_idx:
                indices_to_remove.append(i)
        indices_to_remove.sort(reverse=True)
        for _, item in enumerate(indices_to_remove):
            input_code.writeline(f'del ge_inputs[{item}]')
        for idx, func in enumerate(input_func_list):
            if isinstance(func, _TensorInput):
                continue
            if enable_cache and isinstance(func, _RngStatusInput):
                raise AssertionError(f"Cache compile dose not support operator that depend on RNG, input index: {idx}.")
            input_code.writeline(func.codegen(idx, 'ge_inputs'))
        return input_code.getvalue()

    @staticmethod
    def _gen_input_process(all_input_names, all_sym_input, input_func_list):
        from torch._inductor.utils import IndentedBuffer
        from types import ModuleType
        kernel = IndentedBuffer()
        kernel.writelines(['import torch', 'import numpy', f'def kernel(*args):'])
        with kernel.indent():
            ge_inputs = GeConcreteGraph._codegen_input(all_input_names, None, input_func_list)
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

    @staticmethod
    def _codegen_assert_size_stride(all_meta_tensor_input):
        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()

        for idx, meta in all_meta_tensor_input.items():
            input_code.writelines([f'assert_size_stride(args[{idx}], {tuple(meta.shape)}, {meta.stride()})'])

        return input_code.getvalue()
    

    def _get_current_stream(self):
        if hasattr(self._scope_attr_ctx, 'scope_attr'):
            for attr_dict in self._scope_attr_ctx.scope_attr:
                if "_user_stream_label" in attr_dict:
                    return attr_dict["_user_stream_label"]
        return None

    def _build_record_control_edge(self):
        # find last op before 'air.record.default', and return it's control edge to 'air.wait.default'
        current_stream = self._get_current_stream()
        if self._muti_stream_op_order[current_stream]:
            last_op = self._muti_stream_op_order[current_stream][-1]
            return ControlTensor(last_op).controller
        return 'no_op_before_record'
   
    def _build_wait_control_edge(self, args):
        control_edges = []
        for wait_op_arg in args[0]:
            if isinstance(wait_op_arg, str) and wait_op_arg != 'no_op_before_record':
                control_edges.append(wait_op_arg)
            if isinstance(wait_op_arg, GeTensor):
                control_edges.append(ControlTensor(wait_op_arg.node).controller)
        # push all control edges generated by 'air.wait.default' to stack on current stream        
        current_stream = self._get_current_stream()
        self._wait_control_edges[current_stream].extend(control_edges)
        logger.debug(f"push control edges : {control_edges} generated by 'air.wait.default' to stream: {current_stream}")

    def _handle_wait_control_edge(self, ops):
        if not ops:
            return
        # push ops to a list of it's own stream , in order to help 'air.wait.default' find it's last op
        stream = self._get_current_stream()
        self._muti_stream_op_order[stream].extend(ops)

        if self._wait_control_edges[stream]:
            # pop all control edges generated by 'air.wait.default', then add them to next op after 'air.wait.default', witch is op[0] here
            control_edges = self._wait_control_edges[stream]
            op = ops[0]
            op.input.extend(control_edges)
            logger.info(f"Add record-wait generated control edges: {control_edges} to op: {op.name}")
            self._wait_control_edges[stream] = []     


def _set_dump_options_json(ge_options: Dict[str, Any], ge_graph) -> None:
    """
    Generate dump_options_*.json based on ge_options and torchair.scope.data_dump.
    """
    global g_cached_dump_root

    graph_nodes: List[str] = []
    for op in getattr(ge_graph, "op", []):
        attr_map = getattr(op, "attr", {})
        if "torchair_enable_data_dump" in attr_map:
            val = attr_map["torchair_enable_data_dump"]
            s = getattr(val, "s", val)
            if isinstance(s, (bytes, bytearray)):
                s = s.decode(errors="ignore")
            op_type = getattr(op, "type", None) or getattr(op, "name", None)
            if op_type is not None:
                graph_nodes.append(str(op_type))

    logger.info("Collected graph_nodes for ge data dump: %s", graph_nodes)

    any_present = False
    for key, _ in DUMP_MAPPINGS:
        v = ge_options.get(key)
        if v is not None and str(v).strip():
            any_present = True
            break
    if not any_present:
        logger.info("No dump options found in ge_options; skip writing dump_options.json")
        return

    model_name = getattr(ge_graph, "name", "") or ""
    model_entry: Dict[str, Any] = {"model_name": model_name}

    uniq_layers: Set[str] = set()
    dump_layer_val = ge_options.get("ge.exec.dumpLayer")
    if dump_layer_val:
        for part in (p.strip() for p in str(dump_layer_val).split(" ") if p.strip()):
            uniq_layers.add(part)
    for n in graph_nodes:
        uniq_layers.add(n)
    if uniq_layers:
        model_entry["layer"] = list(uniq_layers)

    if g_cached_dump_root is not None:
        cached = g_cached_dump_root
        if "dump" not in cached or not isinstance(cached["dump"], dict):
            cached["dump"] = {}
        dump_obj = cached["dump"]
        if "dump_list" not in dump_obj or not isinstance(dump_obj["dump_list"], list):
            dump_obj["dump_list"] = []
        dump_obj["dump_list"].append(model_entry)
    else:
        dump_obj: Dict[str, Any] = {}
        for key, mapped in DUMP_MAPPINGS:
            if key == "ge.exec.dumpLayer":
                continue
            value = ge_options.get(key)
            if value is not None and str(value).strip():
                dump_obj[mapped] = str(value)
        dump_obj["dump_list"] = [model_entry]
        g_cached_dump_root = {"dump": dump_obj}

    dir_path = Path.cwd() / "ge_dump_data_json"
    dir_path.mkdir(parents=True, exist_ok=True)

    world_size = 1
    global_rank = 0
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

    out_path = dir_path / f"dump_options_worldsize{world_size}_rank{global_rank}_pid{os.getpid()}.json"
    with out_path.open("w", encoding="utf-8") as ofs:
        json.dump(g_cached_dump_root, ofs, indent=4, ensure_ascii=False)
    logger.info("Dump options written to %s", str(out_path))

    try:
        from torchair.core import _torchair
        _torchair.AclmdlSetDump(str(out_path))
    except Exception as e:
        logger.error("AclmdlSetDump failed for %s: %s", out_path, e)


def _get_ge_dump_options(global_compile_options):
    dump_keys = [
        "ge.exec.enableDump",
        "ge.exec.dumpPath",
        "ge.exec.dumpMode",
        "ge.quant_dumpable",
        "ge.exec.dumpStep",
        "ge.exec.dumpLayer",
        "ge.exec.dumpData",
    ]
    ge_data_dump_options = {
        k: global_compile_options.pop(k)
        for k in dump_keys
        if k in global_compile_options
    }
    return ge_data_dump_options

