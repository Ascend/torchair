from collections import defaultdict
from typing import Dict, List, Set

import sympy
import torch
from npu_extension_for_inductor.common.symbols import Loop, AscExpr
from npu_extension_for_inductor.common.utils import StrRep, TypeUtils
from npu_extension_for_inductor.ir import _Op, _Tensor
from torch._inductor.utils import IndentedBuffer


class ASCGraph:
    def __init__(self, *, name="graph"):
        super().__init__()
        self.name = name
        self._op_count: Dict[str:int] = defaultdict(lambda: 0)
        self.size_vars = set()
        self.axis_vars = dict()
        self.inputs = []
        self.outputs = []
        self.ops: List[_Op] = []
        self.unsupported_ops: Set[str] = set()
        self.fallback_ops: Set[str] = set()
        self.size_vars_holder = self.add_op("Data", name="size_vars")
        self.size_vars_holder.y.dtype = TypeUtils.torch_to_asc(torch.float32)
        self.inputs.append("size_vars")

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

    def add_op(self, type: str, *, name=None, is_unsupported=False):
        if name is None:
            name = type.lower()
            num = self._op_count[name]
            self._op_count[name] += 1
            name = f"{name}{'' if num == 0 else num}"
        op = _Op(type, name)
        op.attr.sched.exec_order = len(self.ops)
        op.attr.sched.axis = sorted(list(self.axis_vars.keys()))
        self.ops.append(op)
        if is_unsupported:
            op.is_unsupported = True
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

    def input(self, name):
        self.inputs.append(name)

    def output(self, name):
        self.outputs.append(name)

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

    def codegen(self, fn_name) -> IndentedBuffer:
        graph = IndentedBuffer()
        graph.writeline(f"def {fn_name}():")
        with graph.indent():
            graph.splice("""
            import os
            if os.getenv('ASCIR_NOT_READY', None) == "1":
                from npu_extension_for_inductor.common.revert_ascir import RevertAscir
                ascir = RevertAscir()
            else:
                from pyautofuse import ascir
            """)  # TODO: remove this once ascir ready
            # Head graph define
            graph.writeline(f"{self.name} = ascir.HintGraph('{self.name}')")
            # Size var and axis
            self.size_vars = sorted(list(self.size_vars))
            for size_var in self.size_vars:
                graph.writeline(f'{size_var} = {self.name}.create_size("{size_var}")')
            for axis, range_expr in self.axis_vars.items():
                graph.writeline(f'{axis} = {self.name}.create_axis("{axis}", {repr(AscExpr(range_expr))})')
            # Ops codegen
            self.size_vars_holder.y.size = [AscExpr(sympy.Symbol(str(s))) for s in self.size_vars]
            for i, op in enumerate(self.ops):
                graph.splice(op.codegen())
            # Inputs and outputs
            graph.splice(f"""
            {self.name}.set_inputs([{', '.join([s for s in self.inputs])}])
            {self.name}.set_outputs([{', '.join([s for s in self.outputs])}])
            """)
            graph.writeline(f"return {self.name}")
        return graph
