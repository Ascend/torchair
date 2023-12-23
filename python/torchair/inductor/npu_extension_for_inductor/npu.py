import dataclasses
import functools
import os
from collections import defaultdict
from typing import List, Iterable, Dict
from sympy import symbols, simplify

import sympy

import torch  # noqa
from torch._inductor.codegen.triton import TritonScheduling

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

    @staticmethod
    def to_dtype(x, dst_dtype, src_dtype=None):
        dst = TypeUtils.torch_to_asc(dst_dtype)
        src = TypeUtils.torch_to_asc(src_dtype)
        return getattr(ir, "to_dtype")(x, dst, src)

    def __getattr__(self, item):
        return getattr(ir, item)


class BufDesc:
    def __init__(self, *, size, dtype, is_output=False, src=None):
        self._size = size
        self._dtype = dtype
        self.is_output: bool = is_output
        self.src = src

    @property
    def asc_size(self):
        return [str(AscExpr(s)) for s in self._size]

    @property
    def asc_dtype(self):
        return TypeUtils.torch_to_asc(self._dtype)


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
        self.buffer.writeline(f"{name}.attr.sched.axis = [{', '.join([s for s in sorted(self.axis_vars)])}]")
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
        if os.getenv('ASCIR_NOT_READY', None) == "1":
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


@dataclasses.dataclass
class Reduction:
    dtype: torch.dtype
    src_dtype: torch.dtype
    reduction_type: str
    value: str
    src: str

    def __str__(self) -> str:
        return self.src


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    @classmethod
    def next_kernel_name(cls):
        name = f"npu_kernel_{cls._index}"
        cls._index += 1
        return name

    def __init__(self, *, var_ranges, indexing_exprs, buf_desc):
        super().__init__()
        self.graph = ASCGraph(self.compute)
        self._buf_desc: Dict[str:BufDesc] = buf_desc
        self._kernel = NPUKernel.next_kernel_name()
        self._size_vars = set()
        self._axis_exprs = dict()
        for axis, expr in var_ranges.items():
            self._size_vars.update(V.graph.sizevars.simplify(expr).free_symbols)
            self._axis_exprs[sympy.Symbol(axis.name)] = expr
        for index, expr in indexing_exprs.items():
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

    def _get_reduce_dims_and_loop(self, index: sympy.Expr):
        reduce_dims = []
        loop = Loop()
        loop.offset = index
        for i, (axis, range) in enumerate(self._axis_exprs.items()):
            stride = index.coeff(axis)
            if str(stride) == "0":
                reduce_dims.append(i)
            else:
                loop.stride.append(stride)
                loop.offset = simplify(loop.offset.subs(axis, 0))
                loop.axis.append(axis)
                loop.size.append(range)
        return reduce_dims, loop

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
        graph_code = self.graph.build()
        from npu_extension_for_inductor.common.debug import draw_asc_graph_dot
        draw_asc_graph_dot(graph_code.getvalue())
        with code.indent():
            code.splice(graph_code)
        code.writeline(
            f"{self._kernel}_compiled = npu_compiler.aclnn(npu_codegen.aclnn({self._kernel}_graph()))")
        code.writeline(f"def {self._kernel}({arg_def}, *, {kw_args_def}):")
        with code.indent():
            code.writeline(f"{self._kernel}_compiled({arg_def}, {kw_args_val})")

        return code.getvalue()

    def load(self, name: str, index: sympy.Expr):
        buf: BufDesc = self._buf_desc[name]
        if not buf.src:
            self.args.input(name)
            self.graph.input(name)
            data = ir.data(name, sizes=buf.asc_size, dtype=buf.asc_dtype)
            buf.src = data
        else:
            data = buf.src
        load = ir.load(data)
        loop = self._index_to_loop(index)
        self.graph.mark_iterable(load, loop)
        if not loop.is_contiguous():
            load = ir.broadcast(load)
            self.graph.mark_iterable(load, loop.contiguous())
        return load

    def _mark_buf_src(self, name, src):
        buf: BufDesc = self._buf_desc[name]
        data = ir.data(name, input=src, sizes=buf.asc_size, dtype=buf.asc_dtype)
        buf.src = data
        if buf.is_output:
            self.args.output(name)
            self.graph.output(name)
        return data

    def store_reduction(self, name, index, value: Reduction):
        reduce_dims, loop = self._get_reduce_dims_and_loop(index)
        reduction = ir.reduction(value.value, dst_dtype=TypeUtils.torch_to_asc(value.dtype),
                                 src_dtype=TypeUtils.torch_to_asc(value.src_dtype), reduce_type=value.reduction_type)
        self.graph.mark_iterable(reduction, loop)
        store = ir.store(reduction)
        self.graph.mark_iterable(store, loop)
        value.src = self._mark_buf_src(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def store(self, name, index, value, mode=None):
        store = ir.store(value)
        self.graph.mark_iterable(store, self._index_to_loop(index))
        self._mark_buf_src(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def reduction(self, dtype, src_dtype, reduction_type, value):
        return Reduction(dtype, src_dtype, reduction_type, value, '')


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self._fuse_judge = TritonScheduling(scheduler)

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_vertical(node1, node2)

    def group_fn(self, sizes):
        return self._fuse_judge.group_fn(sizes)

    def codegen_template(
            self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]
    ):
        raise NotImplementedError()

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        print(f"Codegen for {'Fused' if len(nodes) > 1 else ''}SchedulerNode", flush=True)
        buf_desc = dict()
        var_ranges = None
        indexing_exprs = None
        output_bufs = []
        for i, node in enumerate(nodes):
            body: LoopBody = getattr(node, "_body")
            print(f"Codegen for node {node.node.name}", flush=True)
            print(f"{body.debug_str()}", flush=True)

            if i == 0:
                var_ranges = body.var_ranges
                indexing_exprs = body.indexing_exprs
            else:
                assert str(var_ranges) == str(body.var_ranges)
                assert str(indexing_exprs) == str(body.indexing_exprs)
            is_output = any([isinstance(v.node, torch._inductor.scheduler.OutputNode) for v in node.users])
            if is_output:
                output_bufs.append(node)
            buf_desc[node.node.name] = BufDesc(size=node.node.layout.size, dtype=V.graph.get_dtype(node.node.name),
                                               is_output=is_output)
            for buf in node.read_writes.reads:
                if buf.name not in buf_desc:
                    buf_desc[buf.name] = BufDesc(size=buf.size, dtype=V.graph.get_dtype(buf.name))

        ranges = [[sympy.Symbol(f"{k}")] for k in dict(var_ranges).keys()]
        kernel = NPUKernel(var_ranges=var_ranges, indexing_exprs=indexing_exprs, buf_desc=buf_desc)

        for i, node in enumerate(nodes):
            with V.set_kernel_handler(kernel), kernel:
                node.codegen(ranges)
            kernel.compute.writeline(f"# end of node {node.node.name}")

        wrapper: WrapperCodeGen = V.graph.wrapper_code
        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.code)

        _, call_args, _ = kernel.args.python_argdefs()

        # Manual combine size vars with tensor sizes
        used_sizes = list(sorted(kernel.graph.size_vars))
        call_args.append(f"sym_vals=[{', '.join([s for s in used_sizes])}]")

        for node in output_bufs:
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
        # TODO: implicit assumption that aten or prims ops are supported
        if len(suffix) > 1 and hasattr(ir, suffix[1]):
            return f

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if os.getenv("DISABLE_NPU_FALLBACK", None) == "1":
                return f(*args, **kwargs)
            print(f"Inductor npu fallback: {op}", flush=True)
            return fallback_handler(op, add_to_fallback_set=False)(*args, **kwargs)

        return wrapper

    for op, lower_fn in lowerings.items():
        lowerings[op] = _wrap_npu(op, lower_fn)