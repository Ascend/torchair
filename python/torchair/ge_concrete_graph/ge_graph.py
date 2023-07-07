from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib
import numpy as np
from enum import Enum

import torch
from torch.fx.node import Argument, Target
from torch.utils._mode_utils import no_dispatch

from torchair.core.utils import logger
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, AttrDef, TensorDescriptor
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType


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
    DT_MAX = 33             # Mark the boundaries of data types


def torch_type_to_ge_type(dtype, m=DataType):
    if dtype is None:
        return None
    if dtype == torch.float32:
        return m.DT_FLOAT
    elif dtype == torch.int32:
        return m.DT_INT32
    elif dtype == torch.bool:
        return m.DT_BOOL
    elif dtype == torch.float16:
        return m.DT_FLOAT16
    elif dtype == torch.int8:
        return m.DT_INT8
    elif dtype == torch.uint8:
        return m.DT_UINT8
    elif dtype == torch.int16:
        return m.DT_INT16
    elif dtype == torch.int64:
        return m.DT_INT64
    elif dtype == torch.float64:
        return m.DT_DOUBLE
    else:
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
    if dtype == DataType.DT_INT32:
        return np.int32
    if dtype == DataType.DT_UINT32:
        return np.uint32
    if dtype == DataType.DT_INT64:
        return np.int64
    if dtype == DataType.DT_BOOL:
        return np.bool

    raise ValueError(f"Unsupported ge dtype {dtype}")


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

    raise ValueError(f"Unsupported ge dtype {dtype}")


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
    if dtype == np.int32:
        return DataType.DT_INT32
    if dtype == np.uint32:
        return DataType.DT_UINT32
    if dtype == np.int64:
        return DataType.DT_INT64
    if dtype == np.bool:
        return DataType.DT_BOOL

    raise ValueError(f"Unsupported numpy dtype {dtype}")


def torch_type_to_ge_proto_type(dtype):
    return torch_type_to_ge_type(dtype, ProtoDataType)


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
            self._global_default_graph = GraphDef()
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
            assert isinstance(meta_output, torch.SymInt)
            self._torch_dtype = torch.int64
            self._ge_dtype = DataType.DT_INT64
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
            assert isinstance(meta_output, torch.SymInt)
            self.set_torch_dtype(torch.int64)
            self._symsize = []
            self._desc.attr['_meta'].s = compat_as_bytes(
                f"SymInt({meta_output})")

    def __repr__(self) -> str:
        return f'Tensor({self.tensor}, dtype={_ge_proto_dtype_str(self.desc.dtype)}, size={self._symsize})'


def array_default_f32(v, dtype=None):
    if isinstance(v, float) and dtype is None:
        dtype = np.float32
    return np.array(v, dtype=dtype)


def Const(v: Any, dtype: int = None, name=None) -> Tensor:
    op = get_default_ge_graph().op.add()
    op.type = "Const"
    op.name = next_unique_name(name, "Const")
    value = op.attr["value"].t
    op.attr["_readable_value"].s = compat_as_bytes(str(v))
    if isinstance(v, np.ndarray):
        if dtype is None:
            narray = v
            dtype = _np_dtype_to_ge_dtype(narray.dtype)
        else:
            narray = np.array(v, dtype=_ge_dtype_to_np_dtype(dtype))
    else:
        if dtype is None:
            narray = array_default_f32(v)
            dtype = _np_dtype_to_ge_dtype(narray.dtype)
        else:
            narray = np.array(v, dtype=_ge_dtype_to_np_dtype(dtype))

    value.data = narray.tobytes()
    value.desc.dtype = _ge_dtype_to_ge_proto_dtype(dtype)
    value.desc.layout = "ND"
    value.desc.shape.dim.extend(narray.shape)

    op.output_desc.extend([value.desc])
    return Tensor(op)


def _wrap_ge_tensor(v, dtype=None):
    if isinstance(v, Tensor):
        return v
    return Const(v, dtype=dtype)


def _torch_tensor_to_ge_const(v: torch.Tensor):
    assert type(v) == torch.Tensor
    with no_dispatch():
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
        if not isinstance(input, Tensor):
            if type(input) == torch.Tensor:
                promoted_inputs.append(_torch_tensor_to_ge_const(input))
                continue

            narray = np.array(input)
            assert narray.size > 0, f"Cannot promote {func} input {i} value {input} with no dtype specified"

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
            args = list(args)
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
                        assert optional, f"Input {i} is cannot be None as it is not optional"
                    else:
                        assert isinstance(arg, Tensor)
            return func(*args, **kwargs)
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
