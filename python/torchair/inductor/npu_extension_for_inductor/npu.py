import dataclasses
import functools
import itertools
import os
from typing import List, Iterable, Dict, Union
from unittest.mock import patch

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.symbols import Axis
from npu_extension_for_inductor.common.debug import _left_align_lines, OP_SUMMARY
from sympy import symbols, simplify, Eq

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
from npu_extension_for_inductor.ir import IR as ir, _Tensor, _Scalar


class NPUOverrides(OpOverrides):
    """Map element-wise ops to NPU Triton backend"""

    def __init__(self, parent):
        super().__init__(parent)

    @staticmethod
    def to_dtype(x, dst_dtype, src_dtype=None):
        if dst_dtype == src_dtype:
            return x
        dst = TypeUtils.torch_to_asc(dst_dtype)
        src = TypeUtils.torch_to_asc(src_dtype)
        return ir.cast(x, dst=dst, src=src)

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
        self.size = size
        self.dtype = dtype
        self.is_output: bool = is_output
        self.src = src

    @property
    def asc_size(self):
        return [AscExpr(s) for s in self.size]

    @property
    def asc_dtype(self):
        return TypeUtils.torch_to_asc(self.dtype)


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
    def indirect_indexing(index_var, size, check=False) -> sympy.Symbol:
        kernel: NPUKernel = V.kernel
        return kernel.indirect_indexing(index_var, size, check)


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
        self._indirect_to_scalar: Dict[str, _Scalar] = dict()
        self._size_vars = set()
        self._axis_exprs = dict()
        for axis, expr in var_ranges.items():
            expr = V.graph.sizevars.simplify(expr)
            self._size_vars.update(expr.free_symbols)
            self._axis_exprs[sympy.Symbol(axis.name)] = expr
        for expr in indexing_exprs:
            expr = V.graph.sizevars.simplify(expr)
            self._size_vars.update([s for s in expr.free_symbols if s.name.startswith('s')])
        for desc in self._buf_desc.values():
            desc.size = [V.graph.sizevars.simplify(v) for v in desc.size]
            for size in desc.size:
                self._size_vars.update(size.free_symbols)
        self._size_vars = sorted(self._size_vars, key=lambda x: x.name)
        for size in self._size_vars:
            self.graph.size(size.name)
        self._axis_exprs = dict(sorted(self._axis_exprs.items(), key=lambda item: item[0].name))
        for axis, range_expr in self._axis_exprs.items():
            self.graph.axis(axis.name, range_expr)

        self._contiguous_loop = Loop()
        self._contiguous_loop.axis = list(self._axis_exprs.keys())
        self._contiguous_loop.size = list(self._axis_exprs.values())
        self._contiguous_loop.contiguous_()

    @property
    def contiguous_loop(self):
        return self._contiguous_loop

    def indirect_indexing(self, index_var, size, check=False) -> sympy.Symbol:
        indirect_sym = sympy_symbol(f"npu_scalar{len(self._indirect_to_scalar)}")
        op_name, output_name = str(index_var).split('.')
        src = self.graph.get_op(op_name)
        assert src is not None
        self._indirect_to_scalar[str(indirect_sym)] = _Scalar(_Tensor(getattr(src, output_name)), size, check)
        return indirect_sym

    def __enter__(self):
        super().__enter__()
        assert self.overrides
        self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(NpuCSEProxy(V.get_ops_handler())))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def _buf_size(self, buf):
        return [str(AscExpr(s)) for s in self._buf_desc[buf].size]

    def _buf_dtype(self, buf):
        return TypeUtils.torch_to_asc(self._buf_desc[buf].dtype)

    def _get_reduce_dims_and_loop(self, index: sympy.Expr):
        loop = self._index_to_loop(index)
        reduce_dims = [i for i in range(len(loop.stride)) if str(loop.stride[i]) == "0"]
        return reduce_dims, loop

    def _index_to_loop(self, index: sympy.Expr):
        loop = Loop()
        loop.offset = index
        for axis, range in self._axis_exprs.items():
            loop.stride.append(index.coeff(axis))
            loop.offset = simplify(loop.offset.subs(axis, 0))
            loop.axis.append(axis)
            loop.size.append(sympy.S.One if str(loop.stride[-1]) == "0" else range)
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
        if os.getenv("NPU_INDUCTOR_DUMMY_KERNEL", None) == "1":
            self._kernel_def.writeline(
                "from npu_extension_for_inductor.compiler.aclnn_compiler import DummyNpuInductorKernel")
            self._kernel_def.writeline(f"{self._kernel}_compiled = DummyNpuInductorKernel('{graph_fn}')")
        else:
            self._kernel_def.writeline(f"{self._kernel}_compiled = npu_compiler.aclnn(npu_codegen.aclnn({graph_fn}()))")
        self._kernel_def.writelines(self._comments)
        self._kernel_def.writeline(f"def {self._kernel}({signature_args}):")
        with self._kernel_def.indent():
            self._kernel_def.writeline(f"{self._kernel}_compiled({call_args})")
        code.splice(self._kernel_def)

        return code.getvalue()

    def record_summary(self, nodes, model_path=None):
        labels = [_node_label(node) for node in nodes]
        OP_SUMMARY.add_graph_summary(self.graph, loop='\n'.join(itertools.chain(*labels)), model_path=model_path)

    def view_dot(self, nodes, svg_path=None):
        try:
            import pydot
            dot_graph = self.graph.as_dot()
            labels = [_node_label(node) + ['-' * 20] for node in nodes]
            lines = list(itertools.chain(*labels))
            lines = _left_align_lines(lines)
            dot_graph.add_node(
                pydot.Node(f"{self.graph.name}_body", shape="plaintext", label='\n'.join(lines),
                           fontname="Courier"))
            svg_path = svg_path if svg_path else f"./{self.graph.name}.svg"
            dot_graph.write_svg(svg_path)
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
            if str(s) in self._indirect_to_scalar:
                scalars[s] = self._indirect_to_scalar[str(s)]
        return scalars

    def _get_view_road(self, src: Loop, dst: Loop):
        num_axis = len(src.axis)
        hint_to_axis = []
        for hint, axis, size, order in zip(src.hint_stride, src.axis, src.size, range(num_axis)):
            if hint != 0:
                hint_to_axis.append((hint, Axis(axis, size, order)))
        ordered_axis = [axis for _, axis in sorted(hint_to_axis, reverse=True)]
        non1_order = [axis.order for axis in ordered_axis]
        iter_non1_order = iter(non1_order)
        order = [i if i not in non1_order else next(iter_non1_order) for i in range(num_axis)]

        class MoveOp:
            def __init__(self, *, kind, src, dst):
                self.kind = kind
                self.src = src
                self.dst = dst

        road = []
        dst_loop = dst
        for i, j in zip(range(len(order)), order):
            if i != j:
                src_loop = dst_loop.transpose(i, j).contiguous_()
                src = src.transpose(i, j)
                road.insert(0, MoveOp(kind="transpose", src=src_loop, dst=dst_loop))
                dst_loop = src_loop
                order[i], order[j] = order[j], order[i]

        if [str(v) for v in src.size] != [str(v) for v in dst_loop.size]:
            road.insert(0, MoveOp(kind="broadcast", src=src, dst=dst_loop))

        if len(road) == 0:
            road.append(MoveOp(kind="reinterpret_view", src=src, dst=dst))

        return road

    def load(self, name: str, index: sympy.Expr):
        buf: BufDesc = self._buf_desc[name]
        if not buf.src:
            self.args.input(name)
            self.graph.input(name)
            data = ir.data(name=name, sizes=buf.asc_size, dtype=buf.asc_dtype)
            buf.src = data
        else:
            data = buf.src

        loop = self._index_to_loop(index)
        scalars: Dict[str, _Scalar] = self._get_npu_scalar(index)
        if len(scalars):
            return ir.load_indirect(data.as_loop(loop), *[v.cse for v in scalars.values()], expr=str(index),
                                    syms=[f"{str(k)}={str(v.cse)}(\\<{v.max_value})" for k, v in scalars.items()])
        if loop.is_contiguous():
            load = ir.load(data.as_loop(loop=loop), loop=loop)
        else:
            road = self._get_view_road(loop, self._contiguous_loop)
            loop = road[0].src
            load = ir.load(data.as_loop(loop=loop), loop=loop)
            print(f"Road for index {index} from {loop} to {self.contiguous_loop}", flush=True)
            for op in road:
                print(f"  {op.kind} from {op.src} to {op.dst}", flush=True)
                load = getattr(ir, op.kind)(load, loop=op.dst)
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
        reduction = ir.reduction(value.value, reduce_type=value.reduction_type, loop=loop)
        reduction = NPUOverrides.to_dtype(reduction, dst_dtype=value.dtype, src_dtype=value.src_dtype)

        store = ir.store(reduction, loop=loop)
        value.src = self._mark_buf_src(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def store(self, name, index, value, mode=None):
        store = ir.store(value, loop=self._index_to_loop(index))
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

    def can_fuse_npu(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if len(node1.get_nodes()) == 0 or len(node2.get_nodes()) == 0:
            return True
        body1 = getattr(node1.get_nodes()[0], "_body", None)
        body2 = getattr(node2.get_nodes()[0], "_body", None)
        assert body1 and body2, f"Node {node1.node.name} or {node2.node.name} has no body"
        if str(body1.var_ranges) != str(body2.var_ranges):
            print(f"Cannot fuse {node1.debug_str()} and {node2.debug_str()} due to different var_ranges", flush=True)
            return False
        return True

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if os.getenv("NPU_INDUCTOR_NO_FUSE", None) == "1":
            return False
        return self._fuse_judge.can_fuse_vertical(node1, node2) and self.can_fuse_npu(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if os.getenv("NPU_INDUCTOR_NO_FUSE", None) == "1":
            return False
        return self._fuse_judge.can_fuse_vertical(node1, node2) and self.can_fuse_npu(node1, node2)

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
        indexing_exprs = []
        output_bufs = []
        for i, node in enumerate(nodes):
            body: LoopBody = getattr(node, "_body")

            print(f"Body of {node.node.name}: {body.debug_str()}", flush=True)
            if i == 0:
                var_ranges = body.var_ranges
            else:
                assert str(var_ranges) == str(body.var_ranges), f"{var_ranges} != {body.var_ranges}"
            indexing_exprs += list(body.indexing_exprs.values())
            inner_user_num = sum([user.node in nodes for user in node.users])
            is_output = inner_user_num != len(node.users)
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
            with kernel, kernel.set_current_node(node):
                node.run(*ranges)

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

        from torch._inductor import config
        if config.trace.enabled:
            kernel.benchmark(V.debug.filename(f"{kernel.kernel_name}_benchmark.py"))
            kernel.view_dot(nodes, V.debug.filename(f"{kernel.graph.name}.svg"))
            kernel.record_summary(nodes, V.debug.filename(f"Model.csv"))

        _, call_args, _ = kernel.args.python_argdefs()

        # Manual combine size vars with tensor sizes
        used_sizes = list(sorted(kernel.graph.size_vars))
        call_args.append(f"sym_vals=[{', '.join([str(s) for s in used_sizes])}]")
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
    atexit.register(lambda: OP_SUMMARY.save())

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
            ranges = body.var_ranges
            indexes = list(body.indexing_exprs.values())
            kernel = DummyKernel(var_ranges=ranges, indexing_exprs=indexes, buf_desc=buf_desc)
            kernel.node_to_bounds = body.bounds().get_bounds()

            ranges = [[sympy.Symbol(f"{k}")] for k in dict(body.var_ranges).keys()]

            with kernel:
                body(*ranges)

            if kernel.graph.fallback_reason:
                state.fallback(kernel.graph.fallback_reason)
                return False

            return True

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if os.getenv("NPU_INDUCTOR_ALWAYS_FALLBACK", None) == "1" and isinstance(op, torch._ops.OpOverload):
                print(f"Fallback {op} as: env NPU_INDUCTOR_ALWAYS_FALLBACK=1", flush=True)
                return fallback_handler(op, add_to_fallback_set=False)(*args, **kwargs)

            if "inductor" in str(op):
                return fallback_handler(op, add_to_fallback_set=False)(*args, **kwargs)

            tensor_box = f(*args, **kwargs)

            if os.getenv("NPU_INDUCTOR_DISABLE_FALLBACK", None) == "1":
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
