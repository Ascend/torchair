from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib
import inspect
from enum import Enum
from abc import abstractmethod
import numpy as np

import torch
from torch.fx.node import Argument, Target
from torch.utils._mode_utils import no_dispatch

from torchair.core.utils import logger
from torchair._ge_concrete_graph.ge_ir_pb2 import ModelDef, GraphDef, OpDef, AttrDef, TensorDescriptor
from torchair._ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType


local_variable = threading.local()


class TensorBase():
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def index(self):
        ...

    @abstractmethod
    def dtype(self):
        ...

    @abstractmethod
    def rank(self):
        ...

    __module__ = "torchair.ge"


class TensorSpecBase():
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def dtype(self):
        ...

    @abstractmethod
    def rank(self):
        ...

    @abstractmethod
    def size(self):
        ...

    __module__ = "torchair.ge"


class Format(Enum):
    FORMAT_UNDEFINED = -1
    FORMAT_NCHW = 0
    FORMAT_NHWC = 1
    FORMAT_ND = 2
    FORMAT_NC1HWC0 = 3
    FORMAT_FRACTAL_Z = 4
    FORMAT_NC1HWC0_C04 = 12
    FORMAT_HWCN = 16
    FORMAT_NDHWC = 27
    FORMAT_FRACTAL_NZ = 29
    FORMAT_NCDHW = 30
    FORMAT_NDC1HWC0 = 32
    FORMAT_FRACTAL_Z_3D = 33
    FORMAT_NC = 35
    FORMAT_NCL = 47

    __module__ = "torchair.ge"


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

    __module__ = "torchair.ge"


class TensorType:
    TT_UNKNOWN = 0
    TT_ALL = 1
    TT_QUANTIFIED = 2
    TT_ORDINARY = 3
    TT_BASIC = 4
    TT_NUMBER = 5
    TT_REAL_NUMBER = 6
    TT_COMPLEX = 7
    TT_INTEGER = 8
    TT_SIGNED = 9
    TT_UNSIGNED = 10
    TT_FLOATING = 11
    TT_INDEX_NUMBER = 12
    TT_UNARY = 13
    TT_FLOAT = 14


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
    if dtype == torch.quint4x2:
        return m.DT_INT4

    raise RuntimeError(f"Unsupported torch type {dtype} by ge")


def ge_type_to_torch_type(dtype: DataType) -> torch.dtype:
    if dtype is None:
        return None
    if dtype == DataType.DT_FLOAT:
        return torch.float32
    if dtype == DataType.DT_INT32:
        return torch.int32
    if dtype == DataType.DT_BOOL:
        return torch.bool
    if dtype == DataType.DT_FLOAT16:
        return torch.float16
    if dtype == DataType.DT_INT8:
        return torch.int8
    if dtype == DataType.DT_UINT8:
        return torch.uint8
    if dtype == DataType.DT_INT16:
        return torch.int16
    if dtype == DataType.DT_INT64:
        return torch.int64
    if dtype == DataType.DT_DOUBLE:
        return torch.float64
    if dtype == DataType.DT_BF16:
        return torch.bfloat16
    if dtype == DataType.DT_COMPLEX32:
        return torch.complex32
    if dtype == DataType.DT_COMPLEX64:
        return torch.complex64
    if dtype == DataType.DT_COMPLEX128:
        return torch.complex128
    if dtype == DataType.DT_QINT8:
        return torch.qint8
    if dtype == DataType.DT_QUINT8:
        return torch.quint8
    if dtype == DataType.DT_QINT32:
        return torch.qint32

    raise RuntimeError(f"Unsupported ge type {dtype} by torch")


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


def list_to_torch_dtype(input_list):
    for v in input_list:
        if isinstance(v, torch.SymInt) or isinstance(v, int):
            continue
        else:
            raise RuntimeError(f"Unsupported list type {type(v)}")
    return torch.int64


def torch_type_to_ge_proto_type(dtype):
    return torch_type_to_ge_type(dtype, ProtoDataType)


class _FuncBase:
    def __init__(self):
        pass

    def __call__(self, *args):
        pass

    def codegen(self, ge_idx, ge_inputs_name):
        pass


class _ValueInput(_FuncBase):
    def __init__(self, index: int):
        super().__init__()
        self.fx_input_idx = index

    def __call__(self, *args):
        return torch.tensor(args[self.fx_input_idx])

    def codegen(self, ge_idx, ge_inputs_name):
        return f"{ge_inputs_name}[{ge_idx}] = torch.from_numpy(numpy.array([args[{self.fx_input_idx}]]))"


class _TensorInput(_FuncBase):
    def __init__(self, index: int):
        super().__init__()
        self.fx_input_idx = index

    def __call__(self, *args):
        return args[self.fx_input_idx]

    def codegen(self, ge_idx, ge_inputs_name):
        raise AssertionError("No need to codegen for tensor input")


class _DiscontiguousTensorInput(_FuncBase):
    def __init__(self, index: int):
        super().__init__()
        self.fx_input_idx = index

    def __call__(self, *args):
        return args[self.fx_input_idx].clone()

    def codegen(self, ge_idx, ge_inputs_name):
        return f"{ge_inputs_name}[{ge_idx}] = args[{self.fx_input_idx}].clone()"


class _RngStatusInput(_FuncBase):
    def __init__(self, rng_status):
        super().__init__()
        self.rng_status = rng_status

    def __call__(self, *args):
        offset_input = self.rng_status.consume()
        return offset_input

    def codegen(self, ge_idx, ge_inputs_name):
        return f"{ge_inputs_name}.insert({ge_idx}, None)"


class _SymPackInput(_FuncBase):
    def __init__(self, index_list: List[int]):
        super().__init__()
        self.fx_input_idx_list = index_list

    def __call__(self, *args):
        return torch.tensor([args[idx] for idx in self.fx_input_idx_list])

    def codegen(self, ge_idx, ge_inputs_name):
        input_str = f'{ge_inputs_name}.insert({ge_idx}, torch.from_numpy(numpy.array(['
        for idx in self.fx_input_idx_list:
            input_str += f'args[{idx}], '
        input_str += '])))'
        return input_str


class _ValueType(Enum):
    TENSOR = 0
    PARAMETER = 1
    BUFFER = 2


@dataclass
class _GeInputInfo:
    value_type: _ValueType
    func: _FuncBase
    shape: List[int]
    device_type: str
    dim_gears: Dict[int, List[int]] = field(default_factory=dict)


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
        self._offsets = Data(index=get_default_ge_graph().num_inputs,
                             dtype=DataType.DT_INT64,
                             shape=[1],
                             placement='CPU',
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

    def consume(self):
        offset = self._gen.get_offset()
        self._gen.set_offset(offset + self._offset_count)
        return torch.tensor(self._offset_lists) + offset

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
    def __init__(self, model_def=None, serialized_model_def=None, name=None):
        from torchair.core._backend import TorchNpuGraph
        if model_def is not None and serialized_model_def is not None:
            raise AssertionError(f"Unsupported init method: both model_def and serialized_model_def are specified.")
        elif model_def is None and serialized_model_def is None:
            self._model = ModelDef()
            self._proto = self._model.graph.add()
        elif serialized_model_def is not None:
            self._model = ModelDef()
            self._model.ParseFromString(serialized_model_def)
            self._proto = self._model.graph[0]
        else:
            self._model = model_def
            self._proto = self._model.graph[0]

        self._proto.name = name if name is not None else self._proto.name
        self._executor = TorchNpuGraph(self._proto.name)
        self._python_code = self._python_code_init()
        self._generator_rng_state = defaultdict(map_graph_rng_state)
        self._indexed_inputs = {}
        self._named_inputs_info = {}
        self._used_process_group = {}
        self._dont_prune_me_ops = []


    def _python_code_init(self):
        python_code = ''
        python_code += '# -*- coding: utf-8 -*-\n'
        python_code += 'from torch import tensor\n'
        python_code += 'from torchair._ge_concrete_graph import ge_apis as ge\n'
        python_code += 'from torchair.ge._ge_graph import get_default_ge_graph\n\n'
        return python_code

    def SerializeToString(self):
        return self._model.SerializeToString()

    def load(self, options, *, create_pg=False):
        if create_pg and len(self.used_process_group) != 0:
            rank = torch.distributed.get_rank()
            created_group = {}
            for pg in torch.distributed.distributed_c10d._world.pg_map.keys():
                if torch.distributed.distributed_c10d.get_backend(pg) != "hccl":
                    continue
                hcom_pg_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
                created_group[hcom_pg_name] = hcom_pg_name

            for op in self._proto.op:
                if "group" not in op.attr:
                    continue
                group_name = op.attr["group"].s.decode()
                if group_name not in self.used_process_group:
                    continue
                if group_name not in created_group:
                    rank_list = self.used_process_group[group_name]
                    pg = torch.distributed.distributed_c10d._find_or_create_pg_by_ranks_and_tag(
                        group_name, rank_list, len(rank_list))
                    new_group_name = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
                    created_group[group_name] = new_group_name
                    logger.debug(f'Recover used process group name {group_name} from rank_list {rank_list} '
                                 f'replace it with new_group_name {new_group_name}.')
                op.attr["group"].s = compat_as_bytes(created_group[group_name])

        self._executor.load(self, options)

    def compile(self):
        self._executor.compile()

    def run(self, inputs, assigned_outputs=[], stream=None):
        return self._executor.run(inputs, assigned_outputs, stream)

    def auto_tune(self, example_inputs=[], stream=None):
        self._executor.auto_tune(example_inputs, stream)

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
    def name(self):
        return self._proto.name

    @property
    def generator_rng_state(self):
        return self._generator_rng_state

    @property
    def model(self):
        return self._model

    @name.setter
    def name(self, v):
        self._proto.name = v

    @property
    def num_inputs(self):
        return len(self._indexed_inputs)

    @property
    def indexed_inputs(self):
        return self._indexed_inputs

    @property
    def used_process_group(self):
        return self._used_process_group

    def record_process_group(self, group_name, rank_list, tag):
        self._used_process_group[group_name] = (rank_list, tag)

    def set_used_process_group(self, used_process_group):
        self._used_process_group = used_process_group

    def rng_state(self, philox_num: int = -1, gen: torch.Generator = None):
        _graph_rng_state = self._generator_rng_state[gen]
        seed, offset = _graph_rng_state.next(philox_num)
        input_info = _GeInputInfo(value_type=_ValueType.TENSOR, func=_RngStatusInput(_graph_rng_state),
                                  shape=[len(_graph_rng_state._offset_lists)], device_type="CPU")
        self.record_input_info(_graph_rng_state._offsets.node.name, input_info)

        return seed, offset

    def get_graph_rng_state(self, gen: torch.Generator = None):
        _graph_rng_state = self._generator_rng_state[gen]
        return _graph_rng_state

    def record_input(self, index, op):
        if index in self._indexed_inputs:
            raise AssertionError("index can not in indexed_inputs")
        self._indexed_inputs[index] = op

    def record_input_info(self, name, input_info):
        if name in self._named_inputs_info.keys():
            logger.warning(f'Input info has been recorded repeatedly for data {name}')
        self._named_inputs_info[name] = input_info

    @property
    def must_keep_ops(self):
        return self._dont_prune_me_ops

    @property
    def named_inputs_info(self):
        return self._named_inputs_info

    def dont_prune_me(self, op):
        self._dont_prune_me_ops.append(op)


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


class TensorSpec(TensorSpecBase):
    def __init__(self, meta_output):
        self._meta = meta_output
        if isinstance(meta_output, torch.Tensor):
            self._torch_dtype = meta_output.dtype
            self._ge_dtype = torch_type_to_ge_type(self._torch_dtype)
            self._symsize = list(meta_output.size())
            try:
                self._size = [int(str(s)) for s in self._symsize]
            except Exception:
                self._size = None
        elif is_sym(meta_output):
            self._torch_dtype = sym_to_torch_dtype(meta_output)
            self._ge_dtype = sym_to_ge_dtype(meta_output)
            self._symsize = torch.Size([])
            self._size = []
        else:
            self._torch_dtype = None
            self._ge_dtype = None
            self._symsize = None
            self._size = None

    @property
    def dtype(self):
        return self._ge_dtype

    @property
    def size(self):
        if self._size is None:
            raise AssertionError(f"Trying get size() from dynamic spec {self} is not allowed")
        return self._size

    @property
    def rank(self):
        return len(self._symsize)

    def __repr__(self) -> str:
        return f'TensorSpec(dtype={_ge_proto_dtype_str(_ge_dtype_to_ge_proto_dtype(self._ge_dtype))}, size={self._symsize})'


class ControlTensor:
    def __init__(self, node: OpDef):
        self._node = node

    @property
    def controller(self):
        return f'{self._node.name}:-1'


class Tensor(TensorBase):
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
    def meta(self):
        return self._meta

    @property
    def rank(self):
        if self._symsize is None:
            raise AssertionError(f"Tensor {self} unknown rank")
        return len(self._symsize)

    @property
    def symsize(self):
        return self._symsize

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
        elif isinstance(meta_output, list):
            self.set_torch_dtype(list_to_torch_dtype(meta_output))
            self._symsize = [len(meta_output)]
            self._desc.attr['_meta'].s = compat_as_bytes(
                f"{type(meta_output)}({meta_output})")
        else:
            if not is_sym(meta_output):
                raise AssertionError("meta_output must be one of following instances:"
                "[torch.SymInt, torch.SymFloat, torch.SymBool]")
            self.set_torch_dtype(sym_to_torch_dtype(meta_output))
            self._symsize = []
            self._desc.attr['_meta'].s = compat_as_bytes(
                f"{type(meta_output)}({meta_output})")

    def __repr__(self) -> str:
        return f'Tensor({self.tensor}, dtype={_ge_proto_dtype_str(self.desc.dtype)}, size={self._symsize})'
    
    def get_numel(self):
        if self._symsize is None:
            raise AssertionError(f"Tensor {self} cannot get numel")
        numel = 1
        for sz in self._symsize:
            numel *= sz
        return numel


def get_ge_rng_state(philox_num: int = -1, gen: torch.Generator = None) -> Tuple[int, Tensor]:
    return get_default_ge_graph().rng_state(philox_num, gen)


def dont_prune_me(op: Tensor):
    return get_default_ge_graph().dont_prune_me(op)


def array_default_f32(v, dtype=None):
    if isinstance(v, np.ndarray):
        return v
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
    for k, v in kwargs.items():
        if isinstance(v, str):
            v = f'"{v}"'
        kwargs_list.append(f'{k}={v}')
    return kwargs_list


def _wrap_ge_tensor(v, dtype=None):
    if isinstance(v, Tensor):
        return v
    return Const(v, dtype=dtype)


def _torch_tensor_to_ge_const(v: torch.Tensor):
    if not isinstance(v, torch.Tensor):
        raise AssertionError("input must be instance of torch.Tensor")
    with no_dispatch():
        if v.device.type != "cpu":
            v = v.cpu()
        return Const(v)


def _get_promoted_dtype(inputs: list) -> Tuple[List[DataType], List[DataType]]:
    dtypes = set()
    for input in inputs:
        if isinstance(input, Tensor):
            try:
                dtype = input.dtype
                dtypes.add(dtype)
            except Exception as e:
                logger.debug("Failed to get dtype for input: %s, Error: %s", input, e)

    f_dtypes = []
    i_dtypes = []

    for dtype in dtypes:
        if dtype in [DataType.DT_BF16, DataType.DT_FLOAT16, DataType.DT_FLOAT, DataType.DT_DOUBLE]:
            f_dtypes.append(dtype)
        elif dtype in [DataType.DT_INT8, DataType.DT_UINT8, DataType.DT_INT32, DataType.DT_UINT32, DataType.DT_INT64]:
            i_dtypes.append(dtype)

    return f_dtypes, i_dtypes


def _inputs_to_bundle_inputs(inputs, input_start_end):
    promoted_bundle_inputs = []
    for v in input_start_end:
        if v is None:
            promoted_bundle_inputs.append(None)
        elif isinstance(v, int):
            promoted_bundle_inputs.append(inputs[v])
        else:
            start, end = v
            promoted_bundle_inputs.append(inputs[start:end])
    return promoted_bundle_inputs


def _auto_type_promotion_by_tensor_type(input, inputs_tensor_type, i):
    if (inputs_tensor_type is None) or isinstance(input, Tensor):
        return input
    if inputs_tensor_type[i] == TensorType.TT_INDEX_NUMBER:
        return _wrap_ge_tensor(input, DataType.DT_INT64)
    return input


def _auto_type_promotion_for_const(bundle_inputs: list, inputs_dynamic: list, inputs_optional: list,
                                   inputs_tensor_type: list, func: str) -> list:
    # NetOutput
    if len(inputs_dynamic) == 0:
        return bundle_inputs

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
                inputs.append(_auto_type_promotion_by_tensor_type(input, inputs_tensor_type, i))
        else:
            input_start_end.append(len(inputs))
            inputs.append(_auto_type_promotion_by_tensor_type(input, inputs_tensor_type, i))

    if all([isinstance(input, Tensor) for input in inputs]):
        return _inputs_to_bundle_inputs(inputs, input_start_end)

    f_dtypes, i_dtypes = _get_promoted_dtype(inputs)

    promoted_inputs = []
    for i, input in enumerate(inputs):
        if isinstance(input, Tensor) or (input is None):
            promoted_inputs.append(input)
            continue

        if isinstance(input, torch.Tensor):
            promoted_inputs.append(_torch_tensor_to_ge_const(input))
            continue

        narray = np.array(input)
        if narray.size > 0:
            v = narray.item(0)
            if isinstance(v, float):
                assert_args_checkout(len(f_dtypes) <= 1,
                                     f"Cannot promote {func} input {i} float {v} with dtypes {f_dtypes}")
                promoted_inputs.append(_wrap_ge_tensor(
                    input, dtype=(f_dtypes[0] if f_dtypes else None)))
            elif isinstance(v, int):
                assert_args_checkout(len(i_dtypes) <= 1,
                                             f"Cannot promote {func} input {i} int {v} with dtypes {i_dtypes}")
                promoted_inputs.append(_wrap_ge_tensor(
                    input, dtype=(i_dtypes[0] if i_dtypes else None)))
        else:
            promoted_inputs.append(_wrap_ge_tensor(input))
        logger.debug(
            f"ge.{func} promote input {i} value {input} to dtype {_ge_proto_dtype_str(promoted_inputs[-1].desc.dtype)}")

    return _inputs_to_bundle_inputs(promoted_inputs, input_start_end)


def _set_extral_node_attrs(outputs):
    attr_maps = getattr(local_variable, 'extral_node_attrs', {})
    if len(attr_maps) == 0:
        return

    if isinstance(outputs, Tensor):
        for key, value in attr_maps.items():
            outputs.node.attr[key].s = value
    elif isinstance(outputs, (list, tuple)):
        for output in outputs:
            if not isinstance(output, Tensor):
                raise AssertionError(f'expect tensor, but got {type(output)}.')
            for key, value in attr_maps.items():
                output.node.attr[key].s = value
    else:
        ValueError(f'expect a Tensor, List or Tuple, but got {type(outputs)}.')


def auto_convert_to_tensor(inputs_dynamic, inputs_optional, *, inputs_tensor_type=None):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def dynamic_arg_assert(dynamic, optional, arg):
                if dynamic:
                    if optional:
                        raise AssertionError("Optional input cannot be dynamic")
                    if not isinstance(arg, (list, tuple)):
                        raise AssertionError("args must be a list or a tuple.")
                    if not all([isinstance(v, Tensor) for v in arg]):
                        raise AssertionError("each value of args must be a Tensor.")
                else:
                    if arg is None:
                        if not optional:
                            raise AssertionError(f"Input {i} can not be None as it is not optional")
                    else:
                        if not isinstance(arg, Tensor):
                            raise AssertionError("arg must be a Tensor")

            bundle_inputs = inspect.signature(func).bind(*args, **kwargs)
            args = bundle_inputs.args
            kwargs = bundle_inputs.kwargs

            if len(inputs_dynamic) != len(inputs_optional):
                raise AssertionError("The length of inputs_dynamic must be equal to the length of inputs_optional.")
            if len(args) < len(inputs_dynamic):
                raise AssertionError("The length of args can not be less than inputs_dynamic.")
            args = _auto_type_promotion_for_const(args, inputs_dynamic,
                                                  inputs_optional, inputs_tensor_type, func.__name__)
            for i, dynamic_and_optional in enumerate(zip(inputs_dynamic, inputs_optional)):
                dynamic = dynamic_and_optional[0]
                optional = dynamic_and_optional[1]
                arg = args[i]
                dynamic_arg_assert(dynamic, optional, arg)

            auto_convert_nesting = getattr(local_variable, 'auto_convert_nesting', 0)
            setattr(local_variable, 'auto_convert_nesting', auto_convert_nesting + 1)
            outputs = func(*args, **kwargs)
            local_variable.auto_convert_nesting -= 1

            if local_variable.auto_convert_nesting == 0:
                gegraph = get_default_ge_graph()
                gegraph.add_python_code(bundle_inputs.args, kwargs, outputs, func)

            _set_extral_node_attrs(outputs)

            return outputs
        return wrapper
    return inner


def compat_as_bytes(bytes_or_text, encoding='utf-8'):
    if isinstance(bytes_or_text, str):
        return bytes_or_text.encode(encoding)
    return bytes_or_text


def compat_as_bytes_list(bytes_or_text, encoding='utf-8'):
    if not isinstance(bytes_or_text, (list, tuple)):
        raise AssertionError("bytes_or_text must be a list or a tuple.")
    return [compat_as_bytes(v) for v in bytes_or_text]


def trans_to_list_list_int(lli):
    if not isinstance(lli, (list, tuple)):
        raise AssertionError("lli must be a list or a tuple.")
    attr = AttrDef.ListListInt()
    for li in lli:
        if not isinstance(li, (list, tuple)):
            raise AssertionError("li must be a list or a tuple.")
        list_list_i = attr.list_list_i.add()
        list_list_i.list_i.extend(li)
    return attr


def trans_to_list_list_float(llf):
    if not isinstance(llf, (list, tuple)):
        raise AssertionError("llf must be a list or a tuple.")
    attr = AttrDef.ListListFloat()
    for lf in llf:
        if not isinstance(lf, (list, tuple)):
            raise AssertionError("lf must be a list or a tuple.")
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
    if placement not in ["NPU", "CPU"]:
        raise AssertionError(f"placement should be NPU or CPU, but got {placement}")
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
    if isinstance(v, Tensor):
        raise ValueError("Input v should not be a Tensor object for creating a Const node.")
    if dtype is None:
        if isinstance(v, torch.Tensor):
            dtype = torch_type_to_ge_type(v.dtype)
        else:
            dtype = _np_dtype_to_ge_dtype(array_default_f32(v).dtype)

    target_dtype = ge_type_to_torch_type(dtype)
    origin_v = v
    if not isinstance(v, torch.Tensor):
        with no_dispatch():
            v = torch.tensor(v, dtype=target_dtype)

    if v.dtype != target_dtype:
        v = v.to(target_dtype)
    op = get_default_ge_graph().op.add()
    op.type = "Const"
    op.name = next_unique_name(node_name, "Const")
    value = op.attr["value"].t
    if readable:
        op.attr["_readable_value"].s = compat_as_bytes(f"{repr(origin_v)}")
    import io
    f = io.BytesIO()
    v.untyped_storage()._write_file(f, False, False, torch._utils._element_size(v.dtype))
    value.data = f.getvalue()
    value.desc.dtype = _ge_dtype_to_ge_proto_dtype(dtype)
    value.desc.layout = "ND"
    value.desc.shape.dim.extend(tuple(v.shape))
    op.output_desc.extend([value.desc])
    const_tensor = Tensor(op)

    return const_tensor



@contextlib.contextmanager
def attr_scope(attr_maps):
    try:
        setattr(local_variable, "extral_node_attrs", attr_maps)
        yield
    finally:
        setattr(local_variable, "extral_node_attrs", {})


def assert_args_checkout(arg_flag, message=None):
    if not arg_flag:
        if message is None:
            raise AssertionError
        else:
            raise AssertionError(message)
