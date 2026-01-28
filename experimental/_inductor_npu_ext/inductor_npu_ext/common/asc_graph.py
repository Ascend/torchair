import itertools
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional

import sympy
import torch

from torch._inductor.utils import IndentedBuffer
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges
from inductor_npu_ext.common.symbols import Loop, AscExpr, DenseLoop
from inductor_npu_ext.common.utils import StrRep
from inductor_npu_ext.common.symbols import Loop


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
        super().__init__(v.name, ValueRanges.unknown())
        self.op: _Op = v.parent
        self._v = v

    def as_loop(self, loop: Loop):
        self._v.axis = loop.asc_axis
        self._v.size = loop.asc_size
        self._v.strides = loop.asc_stride
        if loop.offset is not None and str(loop.offset) != "0":
            self._v.offset = loop.asc_offset

        private_name = self.name.replace(f'{self.op.name}.', '')
        if not loop.is_contiguous():
            self.op.set_private_attr(f'not_contiguous', True)
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
    def __init__(self, op, name):
        super().__init__(name)
        self.__dict__['_op'] = op
        self.__dict__['private_attrs'] = {}

    @staticmethod
    def as_asc_attr(v):
        if isinstance(v, torch.dtype):
            return str(v).replace('torch.', 'ascir.dtypes.').replace('bool', 'uint8').replace('bfloat16', 'bf16')
        return repr(v)

    @property
    def op_type(self):
        return self._op

    @property
    def order(self):
        return self.get_private_attr('order')

    @property
    def supported(self):
        return not self.get_private_attr('is_unsupported')

    def get_attr(self, name):
        attr_name = f'{self.name}.{name}'
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        return None

    def get_private_attr(self, name):
        if name in self.private_attrs:
            return self.private_attrs[name]
        return None

    def set_private_attr(self, name, value):
        self.private_attrs[name] = value

    def codegen(self, graph):
        buffer = IndentedBuffer()
        buffer.writeline(f"{self.name} = ascir.ops.{self._op}('{graph}/{self.name}', {graph})")
        for k, v in self.attrs.items():
            buffer.writeline(f"{k} = {self.as_asc_attr(v)}")
        return buffer.getvalue()


class ASCGraph:
    def __init__(self, *, name="graph", hint_str=None):
        super().__init__()
        self.name = name
        self._op_count: Dict[str:int] = defaultdict(lambda: 0)
        self.size_vars = set()
        self.axis_vars = dict()
        self.inputs: List[str] = []
        self.inputs_outer: List[str] = []
        self.outputs: List[str] = []
        self.outputs_outer: List[str] = []
        self.ops: List[_Op] = []
        self.op_cache: Dict[str, Any] = dict()
        self.unsupported_ops: Set[str] = set()
        self.fallback_ops: Set[str] = set()
        self._current_loop: Optional[DenseLoop] = None
        self.hint_str = hint_str

    @property
    def unsupported_reason(self):
        if self.unsupported_ops:
            return f"Unsupported lowered ops: {', '.join(self.unsupported_ops)}"
        return None

    @property
    def fallback_reason(self):
        if self.fallback_ops:
            return f"Must fallback lowered ops: {', '.join(self.fallback_ops)}"
        return None

    @staticmethod
    def is_memory_op(op):
        return op.op_type in ["Data", "Output", "Workspace"]

    def set_current_loop(self, loop: DenseLoop):
        self._current_loop = loop

    def add_op(self, op_type: str, *, name=None, is_unsupported=False):
        if name is None:
            name = op_type.lower()
            num = self._op_count[name]
            self._op_count[name] += 1
            name = f"{name}{'' if num == 0 else num}"
        op = _Op(op_type, name)
        op.set_private_attr("order", len(self.ops))
        if not ASCGraph.is_memory_op(op):
            op.attr.sched.axis = self._current_loop.axis
        self.ops.append(op)
        buffer_name = name
        if op_type != "Data" and hasattr(V.kernel, "current_node") and V.kernel.current_node:
            buffer_name = V.kernel.current_node.node.name
        op.set_private_attr("buffer_name", buffer_name)
        if is_unsupported:
            op.set_private_attr("is_unsupported", True)
            self.unsupported_ops.add(op.op_type)
        return self.ops[-1]

    def add_fallback_op(self, op_type: str, *, name=None):
        self.add_op(op_type, name=name, is_unsupported=True)
        self.fallback_ops.add(op_type)
        return self.ops[-1]

    def get_op(self, name):
        for op in self.ops:
            if op.name == name:
                return op
        return None

    def get_tensor(self, name, index=0):
        op: _Op = self.get_op(name)
        if op is not None:
            if index != 0:
                raise RuntimeError(f"Only support single tensor for now, but got {index}")
            return _Tensor(op.y)
        return None

    def get_input_tensor(self, name):
        try:
            index = self.inputs.index(name)
            return self.get_tensor(f'data{index if index > 0 else ""}')
        except ValueError:
            return None

    def input(self, name, dtype, *, outer_name=None):
        outer_name = outer_name or name
        self.inputs.append(name)
        self.inputs_outer.append(outer_name)
        from inductor_npu_ext import asc_ops as ir
        return ir.data(name=name, dtype=dtype, index=len(self.inputs) - 1)

    def output(self, name, dtype, *, src, outer_name=None):
        outer_name = outer_name or name
        self.outputs.append(name)
        self.outputs_outer.append(outer_name)
        from inductor_npu_ext import asc_ops as ir
        return ir.output(name=name, dtype=dtype, src=src, index=len(self.outputs) - 1)

    def size(self, name):
        self.size_vars.add(StrRep(name))

    def axis(self, name, range_expr):
        self.axis_vars[StrRep(name)] = range_expr

    def as_dot(self):
        from inductor_npu_ext.common.debug import make_graph_dot
        return make_graph_dot(self)

    def codegen(self, var_name=None, with_hint=False) -> IndentedBuffer:
        var_name = var_name or self.name
        graph = IndentedBuffer()
        # Head graph define
        if with_hint and self.hint_str is not None:
            graph.writeline(f"'''")
            graph.splice(self.hint_str)
            graph.writeline(f"'''")
        graph.writeline(f"{var_name} = ascir.HintGraph('{var_name}')")
        # Size var and axis
        self.size_vars = sorted(list(self.size_vars))
        for size_var in self.size_vars:
            graph.writeline(f'{size_var} = {var_name}.create_size("{size_var}")')
        for axis, range_expr in self.axis_vars.items():
            graph.writeline(f'{axis} = {var_name}.create_axis("{axis}", {repr(AscExpr(range_expr))})')
        # Ops codegen
        for i, op in enumerate(self.ops):
            graph.splice(op.codegen(var_name))
        graph.writeline(f"{var_name}.infer_dtypes()")
        return graph


class FusedASCGraph:
    def __init__(self, *, subgraphs: List[ASCGraph], outputs: List[str]):
        super().__init__()
        self._subgraphs: List[ASCGraph] = subgraphs
        buffer_writes = sum([g.outputs for g in subgraphs], [])
        buffer_reads = sum([g.inputs for g in subgraphs], [])

        self.inputs: List[str] = [buf for buf in list(dict.fromkeys(
            buffer_reads)) if buf not in buffer_writes]  # kernel输入地址
        self.outputs: List[str] = outputs  # kernel输出地址
        # Tiling使用的symbols，与外部传参的顺序一致
        self.size_vars = sorted(set(sum([list(g.size_vars) for g in subgraphs], [])))

        self.inputs_outer: List[str] = []  # 输入地址对应的外部fbuffer名，多个地址可能对应一个外部buffer
        self.outputs_outer: List[str] = []  # 输出地址对应的外部fbuffer名，多个地址可能对应一个外部buffer
        self.args: List[str] = []  # 外部buffer的传参顺序

        self.cpp_wrapper: Optional[str] = None  # kernel的cpp_wrapper代码，相同的图结构可能对应不同的cpp_wrapper（inplace导致）
        self.asc_graph: Optional[str] = None  # kernel的asc_graph代码
        self.name: Optional[str] = None  # kernel的唯一名称

    @property
    def subgraphs(self):
        return self._subgraphs

    def as_dot(self):
        from inductor_npu_ext.common.debug import make_fused_graph_dot
        return make_fused_graph_dot(self)

    def codegen(self, var_name) -> IndentedBuffer:
        fused_graph = IndentedBuffer()

        for graph in self._subgraphs:
            fused_graph.writeline(f"# {'-' * 20 + graph.name + '-' * 20}")
            graph_def = graph.codegen(f'{graph.name}_hint', with_hint=var_name != "cache_hint")
            fused_graph.splice(graph_def)

        fused_graph.writeline(f"# {'-' * 20 + var_name + '-' * 20}")
        fused_graph.writeline(f"{var_name} = ascir.FusedGraph('{var_name}')")

        # 对FusedGraph中的所有buffer做匿名化处理，保证缓存命中率
        anonymous_buffers = dict()
        for i, read in enumerate(self.inputs):
            anonymous_buffers[read] = f"input{i}"
        for i, write in enumerate(self.outputs):
            anonymous_buffers[write] = f"output{i}"
        workspace_index = 0
        for buffer in itertools.chain(*[sub.inputs + sub.outputs for sub in self.subgraphs]):
            if buffer not in anonymous_buffers:
                anonymous_buffers[buffer] = f"workspace{workspace_index}"
                workspace_index += 1

        def replace_buffer(buffers: List[str]) -> List[str]:
            return [anonymous_buffers.get(buf, buf) for buf in buffers]

        buffer_writers: Dict[str, List[tuple[ASCGraph, int]]] = {}
        for graph in self._subgraphs:
            fused_graph.writeline(
                f"{graph.name} = ascir.ops.AscBackend('{graph.name}', {graph.name}_hint, {var_name})")
            for i, buffer in enumerate(replace_buffer(graph.outputs)):
                buffer_writers.setdefault(buffer, [])
                buffer_writers[buffer].append((graph, i))

        for i, buffer in enumerate(replace_buffer(self.inputs)):
            fused_graph.writeline(f"{buffer} = ascir.ops.Data('{buffer}', {var_name})")
            fused_graph.writeline(f"{buffer}.attr.ir_attr.index = {i}")

        for buffer, writers in buffer_writers.items():
            if len(writers) > 1:
                fused_graph.writeline(f"{buffer} = [{', '.join([f'{d[0].name}.y[{d[1]}]' for d in writers])}]")
            elif len(writers) == 1:
                fused_graph.writeline(f"{buffer} = {', '.join([f'{d[0].name}.y[{d[1]}]' for d in writers])}")

        for graph in self._subgraphs:
            fused_graph.writeline(f"{graph.name}.x = [{', '.join(replace_buffer(graph.inputs))}]")

        for i, buffer in enumerate(replace_buffer(self.outputs)):
            output_name = f'graph_output{i}'
            fused_graph.writeline(f"{output_name} = ascir.ops.Output('{buffer}', {var_name})")
            fused_graph.writeline(f"{output_name}.attr.ir_attr.index = {i}")
            fused_graph.writeline(f"{output_name}.x = [{buffer}]")

        fused_graph.splice(f'''
        fuser = Autofuser(AutofuserOptions(graph_type=1))
        scheduled_{var_name} = fuser.schedule({var_name})
        tiling_def, host_impl, device_impl = fuser.codegen(scheduled_{var_name})
        ''')

        return fused_graph
