from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib
import inspect
import numpy as np
from enum import Enum

import torch
from torch.fx.node import Argument, Target
from torch.utils._mode_utils import no_dispatch

from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, AttrDef, TensorDescriptor
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType

local_variable = threading.local()

class DataType:
    DT_FLOAT = 0            # float type
    DT_FLOAT16 = 1          # fp16 type
    DT_INT8 = 2             # int8 type
    DT_INT16 = 6            # int16 type
    DT_UINT16 = 7           # uint16 type
    DT_UINT8 = 4            # uint8 type
    DT_INT32 = 3            #
    DT_INT64 = 9            # int64 type
    DT_UINT32 = 8           # unsigned int32
    DT_UINT64 = 10          # unsigned int64
    DT_BOOL = 12            # bool type
    DT_DOUBLE = 11          # double type
    DT_STRING = 13          # string type
    DT_DUAL_SUB_INT8 = 14   # dual output int8 type
    DT_DUAL_SUB_UINT8 = 15  # dual output uint8 type
    DT_COMPLEX64 = 16       # complex64 type
    DT_COMPLEX128 = 17      # complex128 type
    DT_QINT8 = 18           # qint8 type
    DT_QINT16 = 19          # qint16 type
    DT_QINT32 = 20          # qint32 type
    DT_QUINT8 = 21          # quint8 type
    DT_QUINT16 = 22         # quint16 type
    DT_RESOURCE = 23        # resource type
    DT_STRING_REF = 24      # string ref type
    DT_DUAL = 25            # dual output type
    DT_VARIANT = 26         # dt_variant type
    DT_BF16 = 27            # bf16 type
    # Used to indicate a DataType field has not been set.
    DT_UNDEFINED = 28
    DT_INT4 = 29            # int4 type
    DT_UINT1 = 30           # uint1 type
    DT_INT2 = 31            # int2 type
    DT_UINT2 = 32           # uint2 type
    DT_COMPLEX32 = 33       # complex64 type
    DT_MAX = 34             # Mark the boundaries of data types


def torch_type_to_ge_type(dtype, m=DataType):
    if dtype is None:
        return None
    if dtype == torch.float32:
        return m.DT_FLOAT
    if dtype == torch.int32:
        return m.DT_INT32
    if dtype == torch.bool:
        return m.DT_BOOL
    if dtype == torch.float16:
        return m.DT_FLOAT16
    if dtype == torch.int8:
        return m.DT_INT8
    if dtype == torch.uint8:
        return m.DT_UINT8
    if dtype == torch.int16:
        return m.DT_INT16
    if dtype == torch.int64:
        return m.DT_INT64
    if dtype == torch.float64:
        return m.DT_DOUBLE
    if dtype == torch.bfloat16:
        return m.DT_BF16
    if dtype == torch.complex32:
        return m.DT_COMPLEX32
    if dtype == torch.complex64:
        return m.DT_COMPLEX64
    if dtype == torch.complex128:
        return m.DT_COMPLEX128
    if dtype == torch.qint8:
        return m.DT_QINT8
    if dtype == torch.quint8:
        return m.DT_QUINT8
    if dtype == torch.qint32:
        return m.DT_QINT32

    raise RuntimeError(f"Unsupported torch type {dtype} by ge")


def _ge_dtype_to_np_dtype(dtype: DataType) -> np.dtype:
    if dtype == DataType.DT_FLOAT16:
        return np.float16
    if dtype == DataType.DT_FLOAT:
        return np.float32
    if dtype == DataType.DT_DOUBLE:
        return np.float64
    if dtype == DataType.DT_INT8:
        return np.int8
    if dtype == DataType.DT_UINT8:
        return np.uint8
    if dtype == DataType.DT_INT16:
        return np.int16
    if dtype == DataType.DT_UINT16:
        return np.uint16
    if dtype == DataType.DT_INT32:
        return np.int32
    if dtype == DataType.DT_UINT32:
        return np.uint32
    if dtype == DataType.DT_INT64:
        return np.int64
    if dtype == DataType.DT_UINT64:
        return np.uint64
    if dtype == DataType.DT_BOOL:
        return np.bool_
    if dtype == DataType.DT_COMPLEX64:
        return np.complex64
    if dtype == DataType.DT_COMPLEX128:
        return np.complex128

    raise ValueError(f"Unsupported ge dtype {dtype}")


def _is_supported_ge_dtype_by_numpy(ge_dtype: DataType) -> bool:
    try:
        _ge_dtype_to_np_dtype(ge_dtype)
    except Exception:
        return False

    return True


def _ge_dtype_to_ge_proto_dtype(dtype: DataType) -> np.dtype:
    if dtype == DataType.DT_FLOAT16:
        return ProtoDataType.DT_FLOAT16
    if dtype == DataType.DT_FLOAT:
        return ProtoDataType.DT_FLOAT
    if dtype == DataType.DT_DOUBLE:
        return ProtoDataType.DT_DOUBLE
    if dtype == DataType.DT_INT8:
        return ProtoDataType.DT_INT8
    if dtype == DataType.DT_UINT8:
        return ProtoDataType.DT_UINT8
    if dtype == DataType.DT_INT32:
        return ProtoDataType.DT_INT32
    if dtype == DataType.DT_UINT32:
        return ProtoDataType.DT_UINT32
    if dtype == DataType.DT_INT64:
        return ProtoDataType.DT_INT64
    if dtype == DataType.DT_BOOL:
        return ProtoDataType.DT_BOOL
    if dtype == DataType.DT_BF16:
        return ProtoDataType.DT_BF16
    if dtype == DataType.DT_INT16:
        return ProtoDataType.DT_INT16
    if dtype == DataType.DT_COMPLEX32:
        return ProtoDataType.DT_COMPLEX32
    if dtype == DataType.DT_COMPLEX64:
        return ProtoDataType.DT_COMPLEX64
    if dtype == DataType.DT_COMPLEX128:
        return ProtoDataType.DT_COMPLEX128
    if dtype == DataType.DT_QINT8:
        return ProtoDataType.DT_QINT8
    if dtype == DataType.DT_QUINT8:
        return ProtoDataType.DT_QUINT8
    if dtype == DataType.DT_QINT32:
        return ProtoDataType.DT_QINT32

    raise ValueError(f"Unsupported ge dtype {dtype}")


def _ge_proto_dtype_to_ge_dtype(dtype: ProtoDataType):
    if dtype == ProtoDataType.DT_FLOAT16:
        return DataType.DT_FLOAT16
    if dtype == ProtoDataType.DT_FLOAT:
        return DataType.DT_FLOAT
    if dtype == ProtoDataType.DT_DOUBLE:
        return DataType.DT_DOUBLE
    if dtype == ProtoDataType.DT_INT8:
        return DataType.DT_INT8
    if dtype == ProtoDataType.DT_UINT8:
        return DataType.DT_UINT8
    if dtype == ProtoDataType.DT_INT32:
        return DataType.DT_INT32
    if dtype == ProtoDataType.DT_UINT32:
        return DataType.DT_UINT32
    if dtype == ProtoDataType.DT_INT64:
        return DataType.DT_INT64
    if dtype == ProtoDataType.DT_BOOL:
        return DataType.DT_BOOL
    if dtype == ProtoDataType.DT_BF16:
        return DataType.DT_BF16
    if dtype == ProtoDataType.DT_INT16:
        return DataType.DT_INT16
    if dtype == ProtoDataType.DT_COMPLEX32:
        return DataType.DT_COMPLEX32
    if dtype == ProtoDataType.DT_COMPLEX64:
        return DataType.DT_COMPLEX64
    if dtype == ProtoDataType.DT_COMPLEX128:
        return DataType.DT_COMPLEX128
    if dtype == ProtoDataType.DT_QINT8:
        return DataType.DT_QINT8
    if dtype == ProtoDataType.DT_QUINT8:
        return DataType.DT_QUINT8
    if dtype == ProtoDataType.DT_QINT32:
        return DataType.DT_QINT32

    raise ValueError(f"Unsupported ge proto dtype {dtype}")


def _ge_proto_dtype_str(dtype: ProtoDataType) -> str:
    if dtype == ProtoDataType.DT_FLOAT16:
        return "DT_FLOAT16"
    if dtype == ProtoDataType.DT_FLOAT:
        return "DT_FLOAT"
    if dtype == ProtoDataType.DT_DOUBLE:
        return "DT_DOUBLE"
    if dtype == ProtoDataType.DT_INT8:
        return "DT_INT8"
    if dtype == ProtoDataType.DT_UINT8:
        return "DT_UINT8"
    if dtype == ProtoDataType.DT_INT32:
        return "DT_INT32"
    if dtype == ProtoDataType.DT_UINT32:
        return "DT_UINT32"
    if dtype == ProtoDataType.DT_INT64:
        return "DT_INT64"
    if dtype == ProtoDataType.DT_BOOL:
        return "DT_BOOL"
    if dtype == ProtoDataType.DT_BF16:
        return "DT_BF16"

    return "Unknown"


def _np_dtype_to_ge_dtype(dtype: np.dtype) -> ProtoDataType:
    if dtype == np.float16:
        return DataType.DT_FLOAT16
    if dtype == np.float32:
        return DataType.DT_FLOAT
    if dtype == np.float64:
        return DataType.DT_DOUBLE
    if dtype == np.int8:
        return DataType.DT_INT8
    if dtype == np.uint8:
        return DataType.DT_UINT8
    if dtype == np.int16:
        return DataType.DT_INT16
    if dtype == np.uint16:
        return DataType.DT_UINT16
    if dtype == np.int32:
        return DataType.DT_INT32
    if dtype == np.uint32:
        return DataType.DT_UINT32
    if dtype == np.int64:
        return DataType.DT_INT64
    if dtype == np.uint64:
        return DataType.DT_UINT64
    if dtype == np.bool_:
        return DataType.DT_BOOL
    if dtype == np.complex64:
        return DataType.DT_COMPLEX64
    if dtype == np.complex128:
        return DataType.DT_COMPLEX128

    raise ValueError(f"Unsupported numpy dtype {dtype}")


def is_sym(v):
    return isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool))


def sym_to_ge_proto_dtype(v):
    if isinstance(v, torch.SymInt):
        return ProtoDataType.DT_INT64
    if isinstance(v, torch.SymFloat):
        return ProtoDataType.DT_FLOAT
    if isinstance(v, torch.SymBool):
        return ProtoDataType.DT_BOOL
    raise RuntimeError(f"Unsupported sym type {type(v)}")


def sym_to_ge_dtype(v):
    if isinstance(v, torch.SymInt):
        return DataType.DT_INT64
    if isinstance(v, torch.SymFloat):
        return DataType.DT_FLOAT
    if isinstance(v, torch.SymBool):
        return DataType.DT_BOOL
    raise RuntimeError(f"Unsupported sym type {type(v)}")


def sym_to_torch_dtype(v):
    if isinstance(v, torch.SymInt):
        return torch.int64
    if isinstance(v, torch.SymFloat):
        return torch.float32
    if isinstance(v, torch.SymBool):
        return torch.bool
    raise RuntimeError(f"Unsupported sym type {type(v)}")


def torch_type_to_ge_proto_type(dtype):
    return torch_type_to_ge_type(dtype, ProtoDataType)


class _GraphRngState:
    def __init__(self, gen: torch.Generator = None) -> None:
        self._gen = gen
        self._consumed = 0
        if self._gen is None:
            if hasattr(torch, 'npu'):
                idx = torch.npu.current_device()
                self._gen = torch.npu.default_generators[idx]
            else:
                # for st test on cpu:
                torch.manual_seed(10)
                self._gen = torch.default_generator
        self._seed = Const(self._gen.initial_seed(),
                            dtype=DataType.DT_INT64,
                            node_name='initial_seed')
        self._feed_index = get_default_ge_graph().num_inputs()
        self._offsets = Data(index=self._feed_index,
                            dtype=DataType.DT_INT64,
                            shape=[1],
                            placement='NPU',
                            node_name='offset_list')
        self._offset_count = 0
        self._offset_lists = []
        self._unpack_offset = get_default_ge_graph().op.add()
        self._unpack_offset.type = "Unpack"
        self._unpack_offset.name = next_unique_name(
            None, 'unpack_generator_offsets')
        self._unpack_offset.input.append(self._offsets.tensor)
        self._unpack_offset.input_desc.add().CopyFrom(self._offsets.desc)
        self._unpack_offset.input_desc[-1].name = "x"


    def get_idx_and_offset(self):
        return self._feed_index, self._offsets

    def consume(self):
        offset = self._gen.get_offset()
        self._gen.set_offset(offset + self._offset_count)
        return self._feed_index, torch.tensor(self._offset_lists) + offset

    def next(self, philox_num: int = -1):
        self._unpack_offset.output_desc.add().name = "y" + str(self._consumed)
        offset = Tensor(self._unpack_offset, self._consumed)
        self._consumed += 1

        self._offsets.node.input_desc[0].shape.dim[0] = self._consumed
        self._offsets.node.output_desc[0].shape.dim[0] = self._consumed

        self._unpack_offset.attr["num"].i = self._consumed
        self._offset_lists.append(self._offset_count)
        self._offset_count += int((philox_num + 3) / 4) * 4
        return self._seed, offset


def map_graph_rng_state(gen: torch.Generator = None):
    return _GraphRngState(gen)


class GeGraph(object):
    def __init__(self):
        self._proto = GraphDef()
        self._python_code = self._python_code_init()
        self._generator_rng_state = defaultdict(
            map_graph_rng_state)
        self._indexed_inputs = {}

    def _python_code_init(self):
        python_code = ''
        python_code += '# -*- coding: utf-8 -*-\n'
        python_code += 'from torch import tensor\n'
        python_code += 'from torchair.ge_concrete_graph import ge_apis as ge\n'
        python_code += 'from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph\n\n'
        return python_code

    def __getattr__(self, v):
        return getattr(self._proto, v)

    def __str__(self):
        return str(self._proto)

    def __enter__(self):
        _graph_stack.stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _graph_stack.stack.remove(self)

    @staticmethod
    def format_python_code(outputs, func_name, ge_node_name, args, kwargs):
        args_string = ', '.join([f'{i}' for i in args])
        kwargs_string = ', '.join([f'{i}' for i in kwargs])
        inputs_string = ", ".join([i for i in [args_string, kwargs_string] if i])
        if ge_node_name is not None:
            return f'{outputs} = {func_name}({inputs_string}, node_name="{ge_node_name}")'
        else:
            return f'{outputs} = {func_name}({inputs_string})'

    def add_python_code(self, args, kwargs, outputs, func):
        args_list = parse_inputs(args)
        kwargs_list = parse_kwargs(kwargs)
        func_name = f"ge.{func.__name__}"

        outputs_name = ', '.join(parse_inputs(outputs, mode='output'))
        if func.__name__ == 'NetOutput':
            outputs_name = f'\n{outputs_name}'

        ge_name = outputs.node.name if isinstance(outputs, Tensor) else None
        # if node_name is in kwargs, no need to add ge_name
        if 'node_name' in kwargs.keys():
            ge_name = None
        self._python_code += f'{self.format_python_code(outputs_name, func_name, ge_name, args_list, kwargs_list)}\n'

    @property
    def python_code(self):
        return self._python_code

    @property
    def generator_rng_state(self):
        return self._generator_rng_state

    def rng_state(self, philox_num: int = -1, gen: torch.Generator = None):
        _graph_rng_state = self._generator_rng_state[gen]
        return _graph_rng_state.next(philox_num)

    def get_graph_rng_state(self, gen: torch.Generator = None):
        _graph_rng_state = self._generator_rng_state[gen]
        return _graph_rng_state

    def record_input(self, index, op):
        assert index not in self._indexed_inputs
        self._indexed_inputs[index] = op

    def num_inputs(self):
        return len(self._indexed_inputs)


class _GeGraphStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super(_GeGraphStack, self).__init__()
        self.stack = []
        self._global_default_graph = None

    def get_default(self):
        """Override that returns a global default if the stack is empty."""
        if self.stack:
            return self.stack[-1]
        elif self._global_default_graph:
            return self._global_default_graph
        else:
            self._global_default_graph = GeGraph()
            return self._global_default_graph

    @contextlib.contextmanager
    def with_default(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            self.stack.remove(default)


_graph_stack = _GeGraphStack()


def get_default_ge_graph():
    global _graph_stack
    return _graph_stack.get_default()


def default_ge_graph(graph):
    return _graph_stack.with_default(graph)


_g_name_dict = dict()


def next_unique_name(name: str, op: str):
    if name is not None:
        return name

    if op not in _g_name_dict:
        _g_name_dict[op] = 0
        return op
    _g_name_dict[op] += 1
    return f'{op}_{_g_name_dict[op]}'


class TensorSpec:
    def __init__(self, meta_output):
        self._meta = meta_output
        if isinstance(meta_output, torch.Tensor):
            self._torch_dtype = meta_output.dtype
            self._ge_dtype = torch_type_to_ge_type(self._torch_dtype)
            self._symsize = list(meta_output.size())
            try:
                self._size = [int(str(s)) for s in self._symsize]
            except:
                self._size = None
        else:
            assert is_sym(meta_output)
            self._torch_dtype = sym_to_torch_dtype(meta_output)
            self._ge_dtype = sym_to_ge_dtype(meta_output)
            self._symsize = torch.Size([])
            self._size = []

    @property
    def dtype(self):
        return self._ge_dtype

    @property
    def size(self):
        assert self._size is not None, f"Trying get size() from dynamic spec {self} is not allowed"
        return self._size

    @property
    def rank(self):
        return len(self._symsize)

    def __repr__(self) -> str:
        return f'TensorSpec(dtype={_ge_proto_dtype_str(_ge_dtype_to_ge_proto_dtype(self._ge_dtype))}, size={self._symsize})'


class Tensor:
    def __init__(self, node: OpDef, index: int = 0):
        self._node = node
        self._index = index
        self._tensor = f'{node.name}:{index}'
        self._desc = node.output_desc[index]
        self._meta = None
        self._symsize = None
        self._torch_dtype = None
        self._ge_dtype = DataType.DT_UNDEFINED

    @property
    def index(self):
        return self._index

    @property
    def node(self):
        return self._node

    @property
    def tensor(self):
        return self._tensor

    @property
    def controller(self):
        return f'{self._node.name}:-1'

    @property
    def dtype(self):
        return self._ge_dtype

    @property
    def desc(self):
        return self._desc

    @property
    def rank(self):
        assert self._symsize is not None, f"Tensor {self} unknown rank"
        return len(self._symsize)

    def set_torch_dtype(self, dtype):
        self._torch_dtype = dtype
        self._ge_dtype = torch_type_to_ge_type(dtype)
        self._desc.dtype = torch_type_to_ge_proto_type(dtype)

    def set_meta(self, meta_output):
        self._meta = meta_output
        if isinstance(meta_output, torch.Tensor):
            self.set_torch_dtype(meta_output.dtype)
            self._symsize = list(meta_output.size())
            self._desc.attr['_meta'].s = compat_as_bytes(
                f"Tensor(dtype={meta_output.dtype}, shape={meta_output.size()}")
        else:
            assert is_sym(meta_output)
            self.set_torch_dtype(sym_to_torch_dtype(meta_output))
            self._symsize = []
            self._desc.attr['_meta'].s = compat_as_bytes(
                f"{type(meta_output)}({meta_output})")

    def __repr__(self) -> str:
        return f'Tensor({self.tensor}, dtype={_ge_proto_dtype_str(self.desc.dtype)}, size={self._symsize})'


def get_ge_rng_state(philox_num: int = -1, gen: torch.Generator = None) -> Tuple[int, Tensor]:
    return get_default_ge_graph().rng_state(philox_num, gen)


def array_default_f32(v, dtype=None):
    if isinstance(v, list) and len(v) == 0 and dtype is None:
        dtype = np.int32
    if isinstance(v, float) and dtype is None:
        dtype = np.float32
    return np.array(v, dtype=dtype)

def list_depth_check(inputs):
    list_depth = 1

    if isinstance(inputs, (tuple, list)):
        for output in inputs:
            if isinstance(output, (tuple, list)):
                list_depth = 2

    if list_depth == 1:
        inputs = [inputs]

    return inputs

def _parse_variables(variables, mode='input'):
    # Due to the interception of regular Tensor formatting in fake mode, 
    # the string method is used here to avoid errors.
    if isinstance(variables, torch.Tensor):
        return str(variables)

    if not isinstance(variables, Tensor):
        return f'{variables}'

    if variables.node.type == 'Const' and mode == 'input':
        variables_value = variables.node.attr['_readable_value'].s.decode()
        variables_name = f'ge.Const({variables_value}, dtype={_ge_proto_dtype_to_ge_dtype(variables.desc.dtype)})'
    else:
        variables_name = f'{variables.tensor.replace(":", "_").replace("/", "_")}'
    return variables_name


def parse_inputs(inputs, mode='input'):
    if inputs is None:
        return [''] if mode == 'input' else ['_']

    if mode == 'output':
        inputs = list_depth_check(inputs)

    inputs_name_list = []
    for tmp_inputs in inputs:
        if isinstance(tmp_inputs, (tuple, list)):
            tmp_inputs_name_list = []
            for tmp_input in tmp_inputs:
                tmp_inputs_name_list.append(_parse_variables(tmp_input))
            inputs_name_list.append(f"[{', '.join([f'{i}' for i in tmp_inputs_name_list])}]")
        else:
            inputs_name_list.append(f'{_parse_variables(tmp_inputs)}')
    return inputs_name_list


def parse_kwargs(kwargs):
    kwargs_list = []
    for k,v in kwargs.items():
        if isinstance(v, str):
            v = f'"{v}"'
        kwargs_list.append(f'{k}={v}')
    return kwargs_list


def _wrap_ge_tensor(v, dtype=None):
    if isinstance(v, Tensor):
        return v
    return Const(v, dtype=dtype)


def _torch_tensor_to_ge_const(v: torch.Tensor):
    assert type(v) == torch.Tensor
    with no_dispatch():
        if v.device.type != "cpu":
            v = v.cpu()
        return Const(v.numpy())


def _get_promoted_dtype(inputs: list) -> Tuple[List[DataType], List[DataType]]:
    dtypes = set()
    for input in inputs:
        if isinstance(input, Tensor):
            try:
                dtype = input.dtype
                dtypes.add(dtype)
            except:
                pass

    f_dtypes = []
    i_dtypes = []

    for dtype in dtypes:
        if dtype in [DataType.DT_BF16, DataType.DT_FLOAT16, DataType.DT_FLOAT, DataType.DT_DOUBLE]:
            f_dtypes.append(dtype)
        elif dtype in [DataType.DT_INT8, DataType.DT_UINT8, DataType.DT_INT32, DataType.DT_UINT32, DataType.DT_INT64]:
            i_dtypes.append(dtype)

    return f_dtypes, i_dtypes


def _auto_type_promotion_for_const(bundle_inputs: list, inputs_dynamic: list, inputs_optional: list, func: str) -> list:
    inputs = []
    input_start_end = []
    for i, dynamic_and_optional in enumerate(zip(inputs_dynamic, inputs_optional)):
        dynamic = dynamic_and_optional[0]
        optional = dynamic_and_optional[1]
        input = bundle_inputs[i]
        if dynamic:
            input_start_end.append((len(inputs), len(inputs) + len(input)))
            inputs.extend(input)
        elif optional:
            if input is None:
                input_start_end.append(None)
            else:
                input_start_end.append(len(inputs))
                inputs.append(input)
        else:
            input_start_end.append(len(inputs))
            inputs.append(input)

    if all([isinstance(input, Tensor) for input in inputs]):
        return bundle_inputs

    f_dtypes, i_dtypes = _get_promoted_dtype(inputs)

    promoted_inputs = []
    for i, input in enumerate(inputs):
        if not isinstance(input, Tensor) and input is not None:
            if type(input) == torch.Tensor:
                promoted_inputs.append(_torch_tensor_to_ge_const(input))
                continue

            narray = np.array(input)
            if narray.size > 0:
                v = narray.item(0)
                if isinstance(v, float):
                    assert len(
                        f_dtypes) <= 1, f"Cannot promote {func} input {i} float {v} with dtypes {f_dtypes}"
                    promoted_inputs.append(_wrap_ge_tensor(
                        input, dtype=(f_dtypes[0] if len(f_dtypes) else None)))
                elif isinstance(v, int):
                    assert len(
                        i_dtypes) <= 1, f"Cannot promote {func} input {i} int {v} with dtypes {i_dtypes}"
                    promoted_inputs.append(_wrap_ge_tensor(
                        input, dtype=(i_dtypes[0] if len(i_dtypes) else None)))
            else:
                promoted_inputs.append(_wrap_ge_tensor(input))
            logger.info(
                f"ge.{func} promote input {i} value {input} to dtype {_ge_proto_dtype_str(promoted_inputs[-1].desc.dtype)}")
        else:
            promoted_inputs.append(input)

    promoted_bundle_inputs = []
    for v in input_start_end:
        if v is None:
            promoted_bundle_inputs.append(None)
        elif isinstance(v, int):
            promoted_bundle_inputs.append(promoted_inputs[v])
        else:
            start, end = v
            promoted_bundle_inputs.append(promoted_inputs[start:end])

    return promoted_bundle_inputs


def auto_convert_to_tensor(inputs_dynamic, inputs_optional):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bundle_inputs = inspect.signature(func).bind(*args, **kwargs)
            args = bundle_inputs.args
            kwargs = bundle_inputs.kwargs

            assert len(inputs_dynamic) == len(inputs_optional)
            assert len(args) >= len(inputs_dynamic)
            args = _auto_type_promotion_for_const(args, inputs_dynamic,
                                                  inputs_optional, func.__name__)
            for i, dynamic_and_optional in enumerate(zip(inputs_dynamic, inputs_optional)):
                dynamic = dynamic_and_optional[0]
                optional = dynamic_and_optional[1]
                arg = args[i]
                if dynamic:
                    assert not optional, "Optional input cannot be dynamic"
                    assert isinstance(arg, (list, tuple))
                    assert all([isinstance(v, Tensor) for v in arg])
                else:
                    if arg is None:
                        assert optional, f"Input {i} can not be None as it is not optional"
                    else:
                        assert isinstance(arg, Tensor)

            auto_convert_nesting = getattr(local_variable, 'auto_convert_nesting', 0)
            setattr(local_variable, 'auto_convert_nesting', auto_convert_nesting + 1)
            outputs = func(*args, **kwargs)
            local_variable.auto_convert_nesting -= 1

            if local_variable.auto_convert_nesting == 0:
                gegraph = get_default_ge_graph()
                gegraph.add_python_code(bundle_inputs.args, kwargs, outputs, func)

            return outputs
        return wrapper
    return inner


def compat_as_bytes(bytes_or_text, encoding='utf-8'):
    if isinstance(bytes_or_text, str):
        return bytes_or_text.encode(encoding)
    return bytes_or_text


def compat_as_bytes_list(bytes_or_text, encoding='utf-8'):
    assert isinstance(bytes_or_text, (list, tuple))
    return [compat_as_bytes(v) for v in bytes_or_text]


def trans_to_list_list_int(lli):
    assert isinstance(lli, (list, tuple))
    attr = AttrDef.ListListInt()
    for li in lli:
        assert isinstance(li, (list, tuple))
        list_list_i = attr.list_list_i.add()
        list_list_i.list_i.extend(li)
    return attr


def trans_to_list_list_float(llf):
    assert isinstance(llf, (list, tuple))
    attr = AttrDef.ListListFloat()
    for lf in llf:
        assert isinstance(lf, (list, tuple))
        list_list_f = attr.list_list_f.add()
        list_list_f.list_f.extend(lf)
    return attr


_invalid_desc = None


def get_invalid_desc():
    global _invalid_desc
    if _invalid_desc is None:
        _invalid_desc = TensorDescriptor()
        _invalid_desc.dtype = ProtoDataType.DT_UNDEFINED
        _invalid_desc.attr['_is_unfed_optional'].i = 1
    return _invalid_desc


@auto_convert_to_tensor([], [])
def Data(*, index: int, dtype: int, shape: List[int] = None, format: str = "ND", placement: str,
         node_name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Data"
    op.name = next_unique_name(node_name, "Data")
    op.attr["index"].i = index

    desc = op.output_desc.add()
    desc.name = "y"
    desc.dtype = _ge_dtype_to_ge_proto_dtype(dtype)
    desc.layout = format
    assert placement in ["NPU", "CPU"], f"placement should be NPU or CPU, but got {placement}"
    desc.device_type = placement
    if shape is not None:
        desc.shape.dim.extend(shape)
    else:
        desc.shape.dim.extend([-2])

    op.input_desc.add().CopyFrom(desc)
    op.input_desc[-1].name = "x"

    get_default_ge_graph().record_input(index, op)

    return Tensor(op)


@auto_convert_to_tensor([False], [False])
def Cast(x: Tensor, *, dst_type: int, dependencies=[], node_name=None) -> Tensor:
    """
    REG_OP(Cast)
        .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                              DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,
                              DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1}))
        .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                               DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8,
                               DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16}))
        .REQUIRED_ATTR(dst_type, Int)
    """

    op = get_default_ge_graph().op.add()
    op.type = "Cast"
    op.name = next_unique_name(node_name, "Cast")

    # process dependices
    for dependency in dependencies:
        op.input.append(dependency.controller)

    # process inputs
    op.input.append(x.tensor)
    op.input_desc.add().CopyFrom(x.desc)
    op.input_desc[-1].name = "x"

    # process attrs
    op.attr["dst_type"].i = dst_type

    # process outputs
    op.output_desc.add().name = "y"
    y = Tensor(op, 0)

    return y


def Const(v: Any, dtype: int = None, node_name=None, readable=True) -> Tensor:
    if dtype is not None and not _is_supported_ge_dtype_by_numpy(dtype) and not dtype is DataType.DT_BF16:
        # TO DO: unsupported dtype cast for numpy, currently resolved by inserting ge.Cast
        return Cast(Const(v, dtype=None, node_name=node_name), dst_type=dtype)

    op = get_default_ge_graph().op.add()
    op.type = "Const"
    op.name = next_unique_name(node_name, "Const")
    value = op.attr["value"].t

    if isinstance(v, torch.Tensor) and v.dtype is torch.bfloat16:
        # TO DO: Maybe other numpy unsupported dtype
        import io
        f = io.BytesIO()
        v.untyped_storage()._write_file(f, False, False, torch._utils._element_size(torch.bfloat16))
        value.data = f.getvalue()
        value.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_BF16)
        value.desc.layout = "ND"
        value.desc.shape.dim.extend(tuple(v.shape))
        op.output_desc.extend([value.desc])
        const_tensor = Tensor(op)
        return const_tensor

    if dtype is None:
        if isinstance(v, np.ndarray):
            narray = v
        else:
            narray = array_default_f32(v)
        const_ge_dtype = _np_dtype_to_ge_dtype(narray.dtype)
    else:
        narray = np.array(v, dtype=_ge_dtype_to_np_dtype(dtype))
        const_ge_dtype = dtype

    if readable:
        if isinstance(v, (np.ndarray, tuple, list)):
            op.attr["_readable_value"].s = compat_as_bytes(f"{narray.tolist()}")
        else:
            op.attr["_readable_value"].s = compat_as_bytes(f"{narray.item()}")

    value.data = narray.tobytes()
    value.desc.dtype = _ge_dtype_to_ge_proto_dtype(const_ge_dtype)
    value.desc.layout = "ND"
    value.desc.shape.dim.extend(narray.shape)

    op.output_desc.extend([value.desc])
    const_tensor = Tensor(op)

    return const_tensor
