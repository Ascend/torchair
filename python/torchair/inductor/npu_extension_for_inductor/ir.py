from typing import Iterable

from npu_extension_for_inductor.common.symbols import Loop
from torch._inductor.codegen.common import CSEVariable
from torch.utils._sympy.value_ranges import ValueRanges

from . import asc_ops


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
        assert isinstance(v.parent, _Op)
        super().__init__(v.name, ValueRanges.unknown())
        self.op: _Op = v.parent

    def as_loop(self, loop: Loop):
        self.op.mark_output_loop(self.name, loop)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class _Op(_Track):
    def __init__(self, type, name):
        super().__init__(name)
        self.__dict__['_op'] = type
        self.__dict__['_loop'] = {}

    def mark_output_loop(self, output, loop):
        self.__dict__['_loop'][output] = loop

    @property
    def op_type(self):
        return self._op

    def codegen(self):
        from torch._inductor.utils import IndentedBuffer
        buffer = IndentedBuffer()
        buffer.writeline(f"{self.name} = ascir.ops.{self._op}('{self.name}')")
        for k, v in self.attrs.items():
            buffer.writeline(f"{k} = {repr(v)}")
        for k, loop in self.__dict__['_loop'].items():
            buffer.writeline(f"{k}.axis = {loop.asc_axis}")
            buffer.writeline(f"{k}.strides = {loop.asc_stride}")
            buffer.writeline(f"{k}.size = {loop.asc_size}")
        return buffer.getvalue()


class _NpuOverride:
    def __init__(self, f, dtype=None):
        self._f = f
        self._dtype = dtype

    def __call__(self, *args, **kwargs):
        tensors = self._f(*args, **kwargs)
        if isinstance(tensors, Iterable):
            return (_Tensor(t) if not isinstance(t, _Tensor) else t for t in tensors)
        return _Tensor(tensors) if not isinstance(tensors, _Tensor) else tensors


class _IRStore:
    def __init__(self):
        pass

    def __getattr__(self, item):
        self.__dict__[item] = _NpuOverride(getattr(asc_ops, item))
        return self.__dict__[item]


IR = _IRStore()