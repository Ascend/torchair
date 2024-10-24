import functools
from typing import Iterable

import sympy
import torch
from npu_extension_for_inductor.common.symbols import Loop
from npu_extension_for_inductor.common.utils import TypeUtils
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from . import asc_ops


class _NpuOverride:
    def __init__(self, op, f, dtype=None):
        self._op = op
        self._f = f
        self._dtype = dtype

    def _set_loop(self, tensors, loop=None):
        if self._op in ["data", "output", "workspace"]:
            return
        loop = loop if loop else V.kernel.contiguous_loop
        tensors = tensors if isinstance(tensors, (list, tuple)) else [tensors]
        for tensor in tensors:
            tensor.as_loop(loop)

    def __call__(self, *args, loop=None, **kwargs):
        graph = V.kernel.graph
        cache_key = f"{self._op}({args},{loop},{kwargs})"
        if cache_key in graph.op_cache:
            return graph.op_cache[cache_key]
        tensors = self._f(*args, **kwargs)
        if isinstance(tensors, (list, tuple)):
            tensors = (_Tensor(t) if not isinstance(t, _Tensor) else t for t in tensors)
            self._set_loop(tensors, loop)
            graph.op_cache[cache_key] = tensors
            return tensors
        tensor = _Tensor(tensors) if not isinstance(tensors, _Tensor) else tensors
        self._set_loop(tensor, loop)
        graph.op_cache[cache_key] = tensor
        return tensor


def underline_to_camelcase(s):
    return ''.join(word.title() for word in s.split('_'))


def unsupported(*args, op_type, **kwargs):
    graph = V.kernel.graph
    op_type = underline_to_camelcase(op_type)
    if op_type in ["LoadIndirect", "IndexExpr"]:
        op = graph.add_fallback_op(op_type)
    else:
        op = graph.add_op(op_type, is_unsupported=True)
    for i, arg in enumerate(args):
        setattr(op, f"x{i}", arg)
    for k, v in kwargs.items():
        setattr(op, k, v)
    return op.y


class _IRStore:
    def __init__(self):
        pass

    def __getattr__(self, item):
        if hasattr(asc_ops, item):
            self.__dict__[item] = _NpuOverride(item, getattr(asc_ops, item))
        else:
            self.__dict__[item] = _NpuOverride(item, functools.partial(unsupported, op_type=item))
        return self.__dict__[item]


IR = _IRStore()


class _Track:
    def __init__(self, name, parent=None):
        self.__dict__["name"] = f"{parent.name}.{name}" if parent else name
        self.__dict__["_parent"] = parent
        self.__dict__["_root"] = parent.root if parent else self
        if not parent:
            self.__dict__["attrs"] = {}

    def __getattr__(self, item):
        self.__dict__[item] = _Track(item, self)
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = _Track(key, self)
        self._root.attrs[self.__dict__[key].name] = value

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        return self._root


class _Tensor(CSEVariable):
    def __init__(self, v: _Track):
        assert isinstance(v, _Track)
        assert isinstance(v.parent, _Op)
        super().__init__(v.name, ValueRanges.unknown())
        self.op: _Op = v.parent
        self._v = v

    @property
    def dtype(self):
        return self.op.attrs[f'{self.name}.dtype']

    def as_loop(self, loop: Loop):
        self._v.axis = loop.asc_axis
        self._v.strides = loop.asc_stride
        self._v.size = loop.asc_size

        private_name = self.name.replace(f'{self.op.name}.', '')
        self.op.set_private_attr(f'{private_name}.size', loop.hint_size)
        self.op.set_private_attr(f'{private_name}.strides', loop.hint_stride)
        self.op.set_private_attr(f'{private_name}.offset', loop.hint_offset)
        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class _Scalar:
    def __init__(self, cse: _Tensor, max_value: sympy.Symbol, check: bool = True):
        self.cse = cse
        self.max_value = max_value
        self.check = check


class _Op(_Track):
    def __init__(self, type, name):
        super().__init__(name)
        self.__dict__['_op'] = type
        self.__dict__['private_attrs'] = {}

    @property
    def op_type(self):
        return self._op

    @property
    def order(self):
        return self.attrs[f'{self.name}.attr.sched.exec_order']

    @property
    def supported(self):
        return not self.get_private_attr('is_unsupported')

    def get_private_attr(self, name):
        if name in self.private_attrs:
            return self.private_attrs[name]
        return None

    def set_private_attr(self, name, value):
        self.private_attrs[name] = value

    def codegen(self):
        from torch._inductor.utils import IndentedBuffer
        buffer = IndentedBuffer()
        buffer.writeline(f"{self.name} = ascir.ops.{self._op}('{self.name}')")
        for k, v in self.attrs.items():
            buffer.writeline(f"{k} = {repr(v)}")
        return buffer.getvalue()
