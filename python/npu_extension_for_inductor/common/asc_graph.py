from collections import defaultdict
from typing import Dict, List, Set, Any, Optional

import logging
import sympy
import torch
from npu_extension_for_inductor.common.symbols import Loop, AscExpr, DenseLoop
from npu_extension_for_inductor.common.utils import StrRep, TypeUtils
from npu_extension_for_inductor.ir import _Op, _Tensor
from npu_extension_for_inductor.ir import IR as ir
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V


class ASCGraph:
    def __init__(self, *, name="graph"):
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

    @property
    def unsupported_reason(self):
        if len(self.unsupported_ops):
            return f"Unsupported lowered ops: {', '.join(self.unsupported_ops)}"
        return None

    @property
    def fallback_reason(self):
        if len(self.fallback_ops):
            return f"Must fallback lowered ops: {', '.join(self.fallback_ops)}"
        return None

    @staticmethod
    def is_memory_op(op):
        return op.op_type in ["Data", "Output", "Workspace"]

    def set_current_loop(self, loop: DenseLoop):
        self._current_loop = loop

    def add_op(self, type: str, *, name=None, is_unsupported=False):
        if name is None:
            name = type.lower()
            num = self._op_count[name]
            self._op_count[name] += 1
            name = f"{name}{'' if num == 0 else num}"
        op = _Op(type, name)
        op.set_private_attr("order", len(self.ops))
        if not ASCGraph.is_memory_op(op):
            op.attr.sched.axis = self._current_loop.axis
        self.ops.append(op)
        buffer_name = name
        if type != "Data" and hasattr(V.kernel, "current_node") and V.kernel.current_node:
            buffer_name = V.kernel.current_node.node.name
        op.set_private_attr("buffer_name", buffer_name)
        if is_unsupported:
            op.set_private_attr("is_unsupported", True)
            self.unsupported_ops.add(op.op_type)
        return self.ops[-1]

    def add_fallback_op(self, type: str, *, name=None):
        self.add_op(type, name=name, is_unsupported=True)
        self.fallback_ops.add(type)
        return self.ops[-1]

    def get_op(self, name):
        for op in self.ops:
            if op.name == name:
                return op
        return None

    def get_tensor(self, name, index=0):
        op: _Op = self.get_op(name)
        if op is not None:
            assert index == 0, f"Only support single tensor for now, but got {index}"
            return _Tensor(op.y)
        return None

    def input(self, name, dtype, *, outer_name=None):
        outer_name = outer_name or name
        self.inputs.append(name)
        self.inputs_outer.append(outer_name)
        return ir.data(name=name, dtype=dtype, index=len(self.inputs) - 1)

    def output(self, name, dtype, *, src, outer_name=None):
        outer_name = outer_name or name
        self.outputs.append(name)
        self.outputs_outer.append(outer_name)
        return ir.output(name=name, dtype=dtype, src=src, index=len(self.outputs) - 1)

    def size(self, name):
        self.size_vars.add(StrRep(name))

    def axis(self, name, range_expr):
        self.axis_vars[StrRep(name)] = range_expr

    def as_dot(self):
        from npu_extension_for_inductor.common.debug import make_graph_dot
        return make_graph_dot(self)

    def codegen(self, var_name=None) -> IndentedBuffer:
        var_name = var_name or self.name
        graph = IndentedBuffer()
        # Head graph define
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
        return graph


class FusedASCGraph:
    def __init__(self, *, subgraphs: List[ASCGraph], outputs: List[str], name=None):
        super().__init__()
        self.name = name or f"Fused_{'_'.join([g.name for g in subgraphs])}"
        self._subgraphs: List[ASCGraph] = subgraphs
        buffer_writes = sum([g.outputs for g in subgraphs], [])
        buffer_reads = sum([g.inputs for g in subgraphs], [])
        self.inputs: List[str] = list(set(buffer_reads) - set(buffer_writes))
        self.inputs_outer: List[str] = self.inputs
        self.outputs: List[str] = list(outputs)
        self.outputs_outer: List[str] = self.outputs
        self.size_vars = sorted(set(sum([list(g.size_vars) for g in subgraphs], [])))

    @property
    def subgraphs(self):
        return self._subgraphs

    def as_dot(self):
        from npu_extension_for_inductor.common.debug import make_fused_graph_dot
        return make_fused_graph_dot(self)

    def codegen(self, var_name=None) -> IndentedBuffer:
        var_name = var_name or self.name
        fused_graph = IndentedBuffer()

        for graph in self._subgraphs:
            fused_graph.writeline(f"# {'-' * 20 + graph.name + '-' * 20}")
            graph_def = graph.codegen(f'{graph.name}_hint')
            fused_graph.splice(graph_def)

        fused_graph.writeline(f"# {'-' * 20 + self.name + '-' * 20}")
        fused_graph.writeline(f"{self.name} = ascir.FusedGraph('{self.name}')")

        buffer_writers: Dict[str, List[tuple[ASCGraph, int]]] = {}
        for graph in self._subgraphs:
            fused_graph.writeline(f"{graph.name} = ascir.ops.AscGraph('{graph.name}', {graph.name}_hint, {self.name})")
            for i, buffer in enumerate(graph.outputs):
                buffer_writers.setdefault(buffer, [])
                buffer_writers[buffer].append((graph, i))

        for i, buffer in enumerate(self.inputs):
            fused_graph.writeline(f"{buffer} = ascir.ops.Data('{buffer}', {self.name})")
            fused_graph.writeline(f"{buffer}.attr.ir_attr.index = {i}")

        for buffer, writers in buffer_writers.items():
            if len(writers) > 1:
                fused_graph.writeline(f"{buffer} = [{', '.join([f'{d[0].name}.y[{d[1]}]' for d in writers])}]")
            elif len(writers) == 1:
                fused_graph.writeline(f"{buffer} = {', '.join([f'{d[0].name}.y[{d[1]}]' for d in writers])}")

        for graph in self._subgraphs:
            fused_graph.writeline(f"{graph.name}.x = [{', '.join(graph.inputs)}]")

        for i, buffer in enumerate(self.outputs):
            output_name = f'{buffer}_output'
            fused_graph.writeline(f"{output_name} = ascir.ops.Output('{buffer}', {self.name})")
            fused_graph.writeline(f"{output_name}.attr.ir_attr.index = {i}")
            fused_graph.writeline(f"{output_name}.x = [{buffer}]")

        return fused_graph
