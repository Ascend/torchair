from collections import defaultdict
from typing import Dict, List, Set, Any, Optional

import sympy
import torch
from npu_extension_for_inductor.common.symbols import Loop, AscExpr, DenseLoop
from npu_extension_for_inductor.common.utils import StrRep, TypeUtils
from npu_extension_for_inductor.ir import _Op, _Tensor
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V


class ASCGraph:
    def __init__(self, *, name="graph"):
        super().__init__()
        self.name = name
        self._op_count: Dict[str:int] = defaultdict(lambda: 0)
        self.size_vars = set()
        self.axis_vars = dict()
        self.inputs = []
        self.inputs_outer = []
        self.outputs = []
        self.outputs_outer = []
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
        op.attr.sched.exec_order = len(self.ops)
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

    def input(self, name, outer_name=None):
        outer_name = outer_name or name
        self.inputs.append(name)
        self.inputs_outer.append(outer_name)

    def output(self, name, outer_name=None):
        outer_name = outer_name or name
        self.outputs.append(name)
        self.outputs_outer.append(outer_name)

    def size(self, name):
        self.size_vars.add(StrRep(name))

    def axis(self, name, range_expr):
        self.axis_vars[StrRep(name)] = range_expr

    def view_dot(self):
        from npu_extension_for_inductor.common.debug import draw_asc_graph_dot
        draw_asc_graph_dot(self, f"./{self.name}.svg")

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
            graph.splice(op.codegen())
        # Inputs and outputs
        graph.splice(f"""
        {var_name}.set_inputs([{', '.join([s for s in self.inputs])}])
        {var_name}.set_outputs([{', '.join([s for s in self.outputs])}])
        """)
        return graph
