import functools
from collections import defaultdict
from collections import namedtuple
from typing import List, Iterable, Dict
from sympy import symbols, simplify

import sympy

import torch  # noqa

from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.ir import LoopBody
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg, Kernel, OpOverrides,
)
from npu_extension_for_inductor.common.symbols import AscExpr, Loop
from npu_extension_for_inductor.common.utils import TypeUtils
from . import ir


class NPUOverrides(OpOverrides):
    """Map element-wise ops to NPU Triton backend"""

    def __getattr__(self, item):
        return getattr(ir, item)


class ASCGraph:
    def __init__(self, buffer: IndentedBuffer, name="graph"):
        super().__init__()
        self.buffer = buffer
        self._name = name
        self._op_count: Dict[str:int] = defaultdict(lambda: 0)
        self.num_ops = 0
        self.size_vars = set()
        self.axis_vars = set()
        self.inputs = []
        self.outputs = []

    def add_op(self, type: str, name=None):
        if name is None:
            name = type.lower()
            num = self._op_count[name]
            self._op_count[name] += 1
            name = f"{name}{'' if num == 0 else num}"
        self.buffer.writeline(f"{name} = ascir.ops.{type}('{name}')")
        self.buffer.writeline(f"{name}.attr.sched.exec_order = {self.num_ops}")
        self.buffer.writeline(f"{name}.attr.sched.axis = [{', '.join([s for s in self.axis_vars])}]")
        self.num_ops += 1
        return name

    def input(self, name):
        self.inputs.append(name)

    def output(self, name):
        self.outputs.append(name)

    def size(self, name):
        self.size_vars.add(name)
        self.buffer.writeline(f'{name} = {self._name}.create_size("{name}")')

    def axis(self, name, range_expr):
        self.axis_vars.add(name)
        self.buffer.writeline(f'{name} = {self._name}.create_axis("{name}", {AscExpr(range_expr)})')

    def mark_iterable(self, buf: str, desc: Loop):
        self.buffer.writeline(f"{buf}.axis = {desc.asc_axis}")
        self.buffer.writeline(f"{buf}.strides = {desc.asc_stride}")
        self.buffer.writeline(f"{buf}.size = {desc.asc_size}")

    def build(self):
        graph = IndentedBuffer()
        graph.splice("""
        if os.getenv('ASCIR_NOT_READY', None):
            return None
        """)  # TODO: remove this once ascir ready
        graph.splice(f"""
        from pyautofuse import ascir
        {self._name} = ascir.HintGraph('{self._name}')
        """)
        graph.splice(self.buffer.getvalue())
        graph.splice(f"""
        graph.set_inputs([{', '.join([s for s in self.inputs])}])
        graph.set_outputs([{', '.join([s for s in self.outputs])}])
        return graph
        """)
        return graph


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    @classmethod
    def next_kernel_name(cls):
        name = f"npu_kernel_{cls._index}"
        cls._index += 1
        return name

    def __init__(self, *, body: LoopBody, buf_desc):
        super().__init__()
        self.graph = ASCGraph(self.compute)
        self._buf_desc = buf_desc
        self._kernel = NPUKernel.next_kernel_name()
        self._size_vars = set()
        self._axis_exprs = dict()
        for axis, expr in body.var_ranges.items():
            self._size_vars.update(V.graph.sizevars.simplify(expr).free_symbols)
            self._axis_exprs[sympy.Symbol(axis.name)] = expr
        for index, expr in body.indexing_exprs.items():
            self._size_vars.update(
                [s for s in V.graph.sizevars.simplify(expr).free_symbols if not s.name.startswith('z')])
        self._size_vars = sorted(self._size_vars, key=lambda x: x.name)
        for size in self._size_vars:
            self.graph.size(size.name)
        self._axis_exprs = dict(sorted(self._axis_exprs.items(), key=lambda item: item[0].name))
        for axis, range_expr in self._axis_exprs.items():
            self.graph.axis(axis.name, range_expr)

    def _buf_size(self, buf):
        return [str(AscExpr(s)) for s in self._buf_desc[buf].size]

    def _buf_dtype(self, buf):
        return TypeUtils.torch_to_asc(self._buf_desc[buf].dtype)

    def _index_to_loop(self, index: sympy.Expr):
        loop = Loop()
        loop.offset = index
        for axis, range in self._axis_exprs.items():
            loop.stride.append(index.coeff(axis))
            loop.offset = simplify(loop.offset.subs(axis, 0))
            loop.axis.append(axis)
            loop.size.append(range)
        return loop

    @property
    def kernel_name(self):
        return self._kernel

    @property
    def code(self):
        code = IndentedBuffer()
        args, _, _ = self.args.python_argdefs()
        arg_def = ', '.join(args)

        kw_args = ['sym_vals']
        kw_args_def = ', '.join(kw_args)
        kw_args_val = ', '.join([f"{v}={v}" for v in kw_args])

        code.writeline(f"def {self._kernel}_graph():")
        with code.indent():
            code.splice(self.graph.build())
        code.writeline(
            f"{self._kernel}_compiled = npu_compiler.aclnn(npu_codegen.aclnn({self._kernel}_graph()))")
        code.writeline(f"def {self._kernel}({arg_def}, *, {kw_args_def}):")
        with code.indent():
            code.writeline(f"{self._kernel}_compiled({arg_def}, {kw_args_val})")

        return code.getvalue()

    def load(self, name: str, index: sympy.Expr):
        self.args.input(name)
        self.graph.input(name)
        data = ir.data(name, sizes=self._buf_size(name), dtype=self._buf_dtype(name))
        load = ir.load(data)
        self.graph.mark_iterable(load, self._index_to_loop(index))
        return load

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def store(self, name, index, value, mode=None):
        self.args.output(name)
        self.graph.output(name)
        store = ir.store(value)
        self.graph.mark_iterable(store, self._index_to_loop(index))
        ir.data(name, input=store, sizes=self._buf_size(name), dtype=self._buf_dtype(name))
        return store

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return False

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return False

    def group_fn(self, sizes):
        # 这个函数是用来处理迭代大小的，例如将[s0, s1, s2]的sizes变换为[s0, s1*s2]
        return sizes

    def codegen_template(
            self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]
    ):
        raise NotImplementedError()

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        print(f"{'-' * 5} codegen_nodes {'-' * 5}")
        for i, node in enumerate(nodes):
            print(f"Node {i}: {type(node).__name__}\n{node._body.debug_str()}")

            BufDesc = namedtuple("BufDesc", ['size', 'dtype'])
            buf_desc = dict()
            buf_desc[node.node.name] = BufDesc(node.node.layout.size, V.graph.get_dtype(node.node.name))
            for buf in node.read_writes.reads:
                buf_desc[buf.name] = BufDesc(buf.size, V.graph.get_dtype(buf.name))

            kernel = NPUKernel(body=node._body, buf_desc=buf_desc)

            ranges = []
            for k, _ in dict(node._body.var_ranges).items():
                ranges.append([sympy.Symbol(f"{k}")])

            with V.set_kernel_handler(kernel), kernel:
                node.codegen(ranges)

            wrapper: WrapperCodeGen = V.graph.wrapper_code
            wrapper.header.splice("\n\n")
            wrapper.header.splice(kernel.code)

            _, call_args, _ = kernel.args.python_argdefs()

            # Manual combine size vars with tensor sizes
            used_sizes = list(sorted(kernel.graph.size_vars))
            call_args.append(f"sym_vals=[{', '.join([s for s in used_sizes])}]")

            node.mark_run()
            wrapper.writeline(wrapper.wrap_kernel_call(kernel.kernel_name, call_args))

    def codegen_sync(self):
        raise NotImplementedError()

    def flush(self):
        pass

    def benchmark_fused_nodes(self, nodes):
        raise NotImplementedError()


class NpuWrapperCodeGen(WrapperCodeGen):
    def __init__(self):
        super().__init__()
        self.header.splice(
            f"""
                from npu_extension_for_inductor import compiler as npu_compiler
                from npu_extension_for_inductor import codegen as npu_codegen
            """
        )


def as_default_inductor_backend():
    from torch._inductor.codegen.common import register_backend_for_device
    register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)

    from torch._inductor.lowering import fallback_handler
    from torch._inductor.lowering import lowerings
    def _wrap_npu(op, f):
        suffix = str(op).split('.')
        if hasattr(ir, suffix[-1]) or (suffix[-1] == "default" and hasattr(ir, suffix[-2])):
            print(f"Inductor npu currently support: {op}", flush=True)
            return f

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            print(f"Inductor npu force fallback: {op}", flush=True)
            return fallback_handler(op, add_to_fallback_set=False)(*args, **kwargs)

        return wrapper

    for op, lower_fn in lowerings.items():
        lowerings[op] = _wrap_npu(op, lower_fn)
