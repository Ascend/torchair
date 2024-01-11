import dataclasses
import functools
import itertools
import os
from typing import List, Iterable, Dict, Union
from unittest.mock import patch

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.debug import _left_align_lines, OP_SUMMARY
from sympy import symbols, simplify

import sympy

import torch  # noqa
from torch._inductor.codegen.triton import TritonScheduling

from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.ir import LoopBody, TensorBox, IRNode, Loops, StorageBox, FlexibleLayout, ComputedBuffer, Pointwise
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from torch._inductor.utils import sympy_symbol, get_kernel_metadata
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg, Kernel, OpOverrides,
)
from npu_extension_for_inductor.common.symbols import AscExpr, Loop
from npu_extension_for_inductor.common.utils import TypeUtils
from npu_extension_for_inductor.ir import IR as ir, _Tensor


class NPUOverrides(OpOverrides):
    """Map element-wise ops to NPU Triton backend"""

    def __init__(self, parent):
        super().__init__(parent)

    @staticmethod
    def to_dtype(x, dst_dtype, src_dtype=None):
        dst = TypeUtils.torch_to_asc(dst_dtype)
        src = TypeUtils.torch_to_asc(src_dtype)
        return ir.to_dtype(x, dst=dst, src=src)

    @staticmethod
    def constant(value, dtype):
        return ir.constant(value=value, dtype=TypeUtils.torch_to_asc(dtype))

    @staticmethod
    def masked(mask, body, other):
        return ir.masked(mask, body(), other)

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
        return [AscExpr(s) for s in self._size]

    @property
    def asc_dtype(self):
        return TypeUtils.torch_to_asc(self._dtype)


@dataclasses.dataclass
class Reduction:
    dtype: torch.dtype
    src_dtype: torch.dtype
    reduction_type: str
    value: str
    src: str

    def __str__(self) -> str:
        return self.src


class NpuCSEProxy:
    def __init__(self, parent):
        self._parent = parent

    def __getattr__(self, item):
        return getattr(self._parent, item)

    @staticmethod
    def indirect_indexing(index_var, size, check=True) -> sympy.Symbol:
        # 这里在处理内存加载进来的index值
        indirect = V.ops._parent.indirect_indexing(index_var, size, check=check)
        kernel: NPUKernel = V.kernel
        return kernel.add_indirect(indirect)


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    @classmethod
    def next_kernel_name(cls):
        name = f"NpuKernel{cls._index}"
        cls._index += 1
        return name

    def __init__(self, *, var_ranges, indexing_exprs, buf_desc, comments=None):
        super().__init__()
        self._buf_desc: Dict[str:BufDesc] = buf_desc
        self.dtype = next(iter(self._buf_desc.values())).asc_dtype
        self._comments: List[str] = comments
        self._kernel = NPUKernel.next_kernel_name()
        self._kernel_def = IndentedBuffer()
        self._graph_def = IndentedBuffer()
        self.graph = ASCGraph(name=f"{self._kernel}Graph")
        self._indirect_to_ces: Dict[str, _Tensor] = dict()
        self._size_vars = set()
        self._axis_exprs = dict()
        for axis, expr in var_ranges.items():
            self._size_vars.update(V.graph.sizevars.simplify(expr).free_symbols)
            self._axis_exprs[sympy.Symbol(axis.name)] = expr
        for index, expr in indexing_exprs.items():
            self._size_vars.update(
                [s for s in V.graph.sizevars.simplify(expr).free_symbols if s.name.startswith('s')])
        self._size_vars = sorted(self._size_vars, key=lambda x: x.name)
        for size in self._size_vars:
            self.graph.size(size.name)
        self._axis_exprs = dict(sorted(self._axis_exprs.items(), key=lambda item: item[0].name))
        for axis, range_expr in self._axis_exprs.items():
            self.graph.axis(axis.name, range_expr)

    def add_indirect(self, scalar_expr: sympy.Expr):
        indirect_sym = sympy_symbol(f"npu_scalar{len(self._indirect_to_ces)}")
        op_name, output_name = str(scalar_expr).split('.')
        src = self.graph.get_op(op_name)
        assert src is not None
        self._indirect_to_ces[str(indirect_sym)] = _Tensor(getattr(src, output_name))
        return indirect_sym

    def __enter__(self):
        super().__enter__()
        assert self.overrides
        handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(NpuCSEProxy(handler._parent)))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

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

    def codegen(self):
        code = IndentedBuffer()
        args, _, _ = self.args.python_argdefs()
        kw_args = ['sym_vals']

        signature_args = ', '.join(args + ["*"] + kw_args)
        call_args = ', '.join(args + [f"{v}={v}" for v in kw_args])

        graph_fn = self.graph.name
        self._graph_def = self.graph.codegen(graph_fn)

        code.splice(self._graph_def)

        self._kernel_def.clear()
        self._kernel_def.writeline(
            f"{self._kernel}_compiled = npu_compiler.aclnn(npu_codegen.aclnn({graph_fn}()))")
        self._kernel_def.writelines(self._comments)
        self._kernel_def.writeline(f"def {self._kernel}({signature_args}):")
        with self._kernel_def.indent():
            self._kernel_def.writeline(f"{self._kernel}_compiled({call_args})")
        code.splice(self._kernel_def)

        return code.getvalue()

    def view_dot(self, nodes):
        try:
            import pydot
            dot_graph = self.graph.as_dot()
            labels = [_node_label(node) + ['-' * 20] for node in nodes]
            lines = list(itertools.chain(*labels))
            lines = _left_align_lines(lines)
            dot_graph.add_node(
                pydot.Node(f"{self.graph.name}_body", shape="plaintext", label='\n'.join(lines),
                           fontname="Courier"))
            dot_graph.write_svg(f"./{self.graph.name}.svg")
        except ImportError:
            print(f"Unable to save dot for kernel {self.kernel_name} as pydot not installed", flush=True)

    def benchmark(self, file_path=None):
        file_path = file_path if file_path else f"./{self._kernel}_benchmark.py"
        if not self._kernel_def.getvalue():
            self.codegen()
        with open(file_path, "w") as f:
            becnhmark_code = IndentedBuffer()
            becnhmark_code.splice(self._graph_def)
            becnhmark_code.writelines(["\n"] * 2)

            becnhmark_code.writeline("if __name__ == '__main__':")
            with becnhmark_code.indent():
                becnhmark_code.splice("""
                import os
                # os.environ['ASCIR_NOT_READY'] = "1"
                os.environ["NPU_CORE_TYPE"] = "ai_core-ascend910B1"
                """)
                becnhmark_code.writeline("# Construct asc graph")
                becnhmark_code.writeline(f"graph = {self.graph.name}()")
                becnhmark_code.splice("""
                # Compile and run npu kernel
                from npu_extension_for_inductor import compiler as npu_compiler
                from npu_extension_for_inductor import codegen as npu_codegen
                """)
                becnhmark_code.splice(self._kernel_def.getvalue().replace(f"{self.graph.name}()", "graph"))
                becnhmark_code.writeline(f"# Add your test code here")
            f.write(becnhmark_code.getvalue())

    def _get_npu_scalar(self, index: sympy.Expr):
        scalars = dict()
        for s in index.free_symbols:
            if str(s) in self._indirect_to_ces:
                scalars[s] = self._indirect_to_ces[str(s)]
        return scalars

    def load(self, name: str, index: sympy.Expr):
        buf: BufDesc = self._buf_desc[name]
        if not buf.src:
            self.args.input(name)
            self.graph.input(name)
            data = ir.data(name=name, sizes=buf.asc_size, dtype=buf.asc_dtype)
            buf.src = data
        else:
            data = buf.src

        scalars = self._get_npu_scalar(index)
        if len(scalars):
            return ir.load_indrect(data, *scalars.values(), expr=str(index), syms=[str(k) for k in scalars.keys()])

        load = ir.load(data)
        loop = self._index_to_loop(index)
        self.graph.mark_iterable(load, loop)
        if not loop.is_contiguous():
            load = ir.broadcast(load)
            self.graph.mark_iterable(load, loop.contiguous())
        return load

    def _mark_buf_src(self, name, src):
        buf: BufDesc = self._buf_desc[name]
        if buf.is_output:
            data = ir.output(name=name, input=src, sizes=buf.asc_size, dtype=buf.asc_dtype)
        else:
            data = ir.workspace(name=name, input=src, sizes=buf.asc_size, dtype=buf.asc_dtype)
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

    def rename_indexing(self, index) -> sympy.Expr:
        if isinstance(index, sympy.Symbol) and index.name.startswith("s"):
            self.graph.size(index.name)
            return index
        return super().rename_indexing(index)

    @property
    def assert_function(self):
        return "ascir.Assert"

    def index_to_str(self, index):
        return str(index)


class DummyKernel(NPUKernel):
    def __init__(self, *args, **kwargs):
        with patch.object(NPUKernel, "next_kernel_name", lambda: "DummyKernel"):
            super().__init__(*args, **kwargs)


def _node_comment(node: Union[BaseSchedulerNode, List[BaseSchedulerNode]]):
    node = node if isinstance(node, (list, tuple)) else [node]
    origin_str, detailed_origin_str = get_kernel_metadata(node, V.graph.wrapper_code)
    lines = []
    if origin_str:
        lines.append(origin_str)
        lines.extend(detailed_origin_str.split("\n"))
    return lines


def _node_label(node: SchedulerNode):
    lines = [f"<Node %{node.node.name}% body>:"]
    lines.extend(_node_comment(node))
    lines.extend(node._body.debug_str().split("\n"))
    lines = [l for l in lines if l]
    return lines


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
                assert str(var_ranges) == str(body.var_ranges), f"{var_ranges} != {body.var_ranges}"
                assert str(indexing_exprs) == str(body.indexing_exprs), f"{indexing_exprs} != {body.indexing_exprs}"
            is_output = any([isinstance(v.node, torch._inductor.scheduler.OutputNode) for v in node.users])
            if is_output:
                output_bufs.append(node)
            buf_desc[node.node.name] = BufDesc(size=node.node.layout.size, dtype=V.graph.get_dtype(node.node.name),
                                               is_output=is_output)
            for buf in node.read_writes.reads:
                if buf.name not in buf_desc:
                    buf_desc[buf.name] = BufDesc(size=buf.size, dtype=V.graph.get_dtype(buf.name))

        ranges = [[sympy.Symbol(f"{k}")] for k in dict(var_ranges).keys()]

        wrapper: WrapperCodeGen = V.graph.wrapper_code
        comments = _node_comment(nodes)
        kernel = NPUKernel(var_ranges=var_ranges, indexing_exprs=indexing_exprs, buf_desc=buf_desc, comments=comments)
        for comment in comments:
            wrapper.writeline(comment)

        for i, node in enumerate(nodes):
            with kernel:
                node.codegen(ranges)

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

        kernel.benchmark()  # TODO: Make this work under config
        kernel.view_dot(nodes)  # TODO: Make this work under config

        _, call_args, _ = kernel.args.python_argdefs()

        # Manual combine size vars with tensor sizes
        used_sizes = list(sorted(kernel.graph.size_vars))
        call_args.append(f"sym_vals=[{', '.join([str(s) for s in used_sizes])}]")

        for node in output_bufs:
            node.mark_run()
        wrapper.writeline(wrapper.wrap_kernel_call(kernel.kernel_name, [str(v) for v in call_args]))

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
    register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)
    import atexit
    atexit.register(lambda: OP_SUMMARY.print())

    from torch._inductor.lowering import fallback_handler
    from torch._inductor.lowering import lowerings
    def _wrap_npu(op, f):
        class _OpState:
            CACHED_STATE: Dict[str, "_OpState"] = dict()

            @classmethod
            def get_cached_state(cls, op, tensor_box) -> "_OpState":
                if op not in cls.CACHED_STATE:
                    state = _OpState(op)
                    with patch.object(FlexibleLayout, "allow_indexing", True):
                        is_supported_box(tensor_box, state)
                    cls.CACHED_STATE[op] = state
                return cls.CACHED_STATE[op]

            def __init__(self, op):
                self.op = op
                self.fallback_reason: str = ""

            def fallback(self, reason: str):
                self.fallback_reason = reason

            @property
            def is_supported(self):
                return not self.fallback_reason

        def is_supported_box(tensor_box, state: _OpState):
            if not isinstance(tensor_box, TensorBox):
                return True
            if not isinstance(tensor_box.data, StorageBox):
                return True
            if not isinstance(tensor_box.data.data, (Pointwise, Reduction)):
                return True

            box: StorageBox = tensor_box.data
            buffer = ComputedBuffer(
                name=str(op),
                layout=FlexibleLayout(
                    device=box.data.get_device(),
                    dtype=box.data.get_dtype(),
                    size=box.data.get_size(),
                ).as_fixed(),
                data=box.data)

            _, body = buffer.simplify_and_reorder()
            buf_desc = dict()
            buf_desc[buffer.name] = BufDesc(size=buffer.layout.size, dtype=box.data.get_dtype(), is_output=True)
            for buf in buffer.get_reads():
                if buf.name not in buf_desc:
                    buf_desc[buf.name] = BufDesc(size=buf.size, dtype=V.graph.get_dtype(buf.name))
            kernel = DummyKernel(var_ranges=body.var_ranges, indexing_exprs=body.indexing_exprs, buf_desc=buf_desc)
            kernel.node_to_bounds = body.bounds().get_bounds()

            ranges = [[sympy.Symbol(f"{k}")] for k in dict(body.var_ranges).keys()]

            with kernel:
                body(*ranges)

            if kernel.graph.unsupported_reason:
                state.fallback(kernel.graph.unsupported_reason)
                return False

            return True

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            tensor_box = f(*args, **kwargs)

            if os.getenv("DISABLE_NPU_FALLBACK", None) == "1":
                return tensor_box

            state: _OpState = _OpState.get_cached_state(op, tensor_box)
            if not state.is_supported:
                print(f"Fallback {op} as: {state.fallback_reason}", flush=True)
                OP_SUMMARY.fallback(op, state.fallback_reason)
                return fallback_handler(op, add_to_fallback_set=False)(*args, **kwargs)

            return tensor_box

        return wrapper

    for op, lower_fn in lowerings.items():
        lowerings[op] = _wrap_npu(op, lower_fn)
