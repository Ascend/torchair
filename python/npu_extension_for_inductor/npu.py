import dataclasses
import functools
import itertools
import contextlib
import logging
import os
from typing import List, Iterable, Dict, Union
from unittest.mock import patch

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.symbols import Axis
from npu_extension_for_inductor.common.debug import _left_align_lines, OP_SUMMARY
from npu_extension_for_inductor.common.utils import camel_to_snake
from sympy import symbols, simplify, Eq

import sympy

import torch  # noqa
from torch._inductor.codegen.triton import TritonScheduling

from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.ir import LoopBody
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode, FusedSchedulerNode
from torch._inductor.utils import sympy_symbol, get_kernel_metadata
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg, Kernel, OpOverrides,
)
from npu_extension_for_inductor.common.symbols import AscExpr, Loop, DenseLoop
from npu_extension_for_inductor.common.utils import TypeUtils
from npu_extension_for_inductor.ir import IR as ir, _Tensor, _Scalar
from npu_extension_for_inductor.ir import UBConcat


class NPUOverrides(OpOverrides):
    """Map element-wise ops to NPU Triton backend"""

    def __init__(self, parent):
        super().__init__(parent)

    def __getattr__(self, item):
        return getattr(ir, item)

    @staticmethod
    def to_dtype(x, dst_dtype, src_dtype=None):
        if dst_dtype == src_dtype:
            return x
        dst = TypeUtils.torch_to_asc(dst_dtype)
        src = TypeUtils.torch_to_asc(src_dtype)
        return ir.cast(x, dst=dst, src=src)

    @staticmethod
    def logical_not(x):
        return ir.logical_not(x)

    @staticmethod
    def constant(value, dtype):
        return ir.constant(value=repr(value))

    @staticmethod
    def masked(mask, body, other):
        return ir.masked(mask, body(), other)

    @staticmethod
    def reciprocal(x):
        return ir.div(ir.constant("1"), x)

    @staticmethod
    def square(x):
        return ir.mul(x, x)

    @staticmethod
    def bitwise_not(x):
        return ir.bitwise_not(x)

    @staticmethod
    def bitwise_and(x, y):
        return ir.bitwise_and(x, y)

    @staticmethod
    def bitwise_or(x, y):
        return ir.bitwise_or(x, y)

    @staticmethod
    def bitwise_xor(x, y):
        return ir.bitwise_xor(x, y)

    @staticmethod
    def bitwise_left_shift(x, y):
        return ir.bitwise_left_shift(x, y)

    @staticmethod
    def bitwise_right_shift(x, y):
        return ir.bitwise_right_shift(x, y)

    @staticmethod
    def load_seed(name, offset):
        return ir.load_seed(offset=sympy.Integer(offset))


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

    def __getitem__(self, index):  # Welford reduction
        return self

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

    def __init__(self, nodes, *, comments=None):
        super().__init__()
        self._buf_desc: Dict[str:BufDesc] = {}
        self._comments: List[str] = comments
        self._kernel = NPUKernel.next_kernel_name()
        self._kernel_def = IndentedBuffer()
        self.graph = ASCGraph(name=f"{self._kernel}Graph")
        self._indirect_to_scalar: Dict[str, _Scalar] = dict()
        self._current_loop = None
        self._current_input_index = 0

        for node in nodes:
            inner_user_num = sum([user.node in nodes for user in node.users])
            is_output = inner_user_num != len(node.users)
            layout_size = [V.graph.sizevars.simplify(s) for s in node.node.layout.size]
            self._buf_desc[node.node.name] = BufDesc(size=layout_size, dtype=V.graph.get_dtype(node.node.name),
                                                     is_output=is_output)
            for buf in node.read_writes.reads:
                if buf.name not in self._buf_desc:
                    self._buf_desc[buf.name] = BufDesc(size=buf.size, dtype=V.graph.get_dtype(buf.name))
        self.dtype = next(iter(self._buf_desc.values())).asc_dtype

    def __enter__(self):
        super().__enter__()
        assert self.overrides
        self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(NpuCSEProxy(V.get_ops_handler())))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    @property
    def contiguous_loop(self):
        return self._current_loop

    @property
    def kernel_name(self):
        return self._kernel

    @property
    def assert_function(self):
        return "ascir.Assert"

    @classmethod
    def next_kernel_name(cls):
        name = f"NpuKernel{cls._index}"
        cls._index += 1
        return name

    @contextlib.contextmanager
    def set_current_loop(self, loop: DenseLoop):
        assert isinstance(loop, DenseLoop)
        self._current_input_index = 0
        self.graph.set_current_loop(loop)
        prior = self._current_loop
        self._current_loop = loop
        try:
            yield
        finally:
            self._current_loop = prior

    def indirect_indexing(self, index_var, size, check=False) -> sympy.Symbol:
        indirect_sym = sympy_symbol(f"npu_scalar{len(self._indirect_to_scalar)}")
        op_name, output_name = str(index_var).split('.')
        src = self.graph.get_op(op_name)
        assert src is not None
        self._indirect_to_scalar[str(indirect_sym)] = _Scalar(_Tensor(getattr(src, output_name)), size, check)
        return indirect_sym

    def codegen(self):
        code = IndentedBuffer()
        args, _, _ = self.args.python_argdefs()
        call_args = sorted(args)
        args.append('workspace')
        call_args.append('workspace')
        kw_args = [str(v) for v in list(sorted(self.graph.size_vars))]

        signature_args = ', '.join(args + ["*"] + kw_args) if len(kw_args) else ', '.join(args)
        call_args = ', '.join(call_args + [f"{v}={v}" for v in kw_args])

        self._kernel_def.clear()
        kernel_var_name = f"{self._kernel}_compiled"

        from npu_extension_for_inductor import codegen as npu_codegen
        self._kernel_def.splice(npu_codegen.codegen_kernel_def(self.graph, kernel_var_name))
        self._kernel_def.writelines(self._comments)
        self._kernel_def.writeline(f"def {self._kernel}({signature_args}):")
        with self._kernel_def.indent():
            self._kernel_def.writeline(f"{kernel_var_name}({call_args})")
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
            logging.info(f"Unable to save dot for kernel {self.kernel_name} as pydot not installed")

    def benchmark(self, file_path=None):
        file_path = file_path if file_path else f"./{self._kernel}_benchmark.py"
        if not self._kernel_def.getvalue():
            self.codegen()
        with open(file_path, "w") as f:
            becnhmark_code = IndentedBuffer()
            becnhmark_code.writeline("from npu_extension_for_inductor import compiler as npu_compiler")
            becnhmark_code.splice(self._kernel_def)
            becnhmark_code.writelines(["\n"] * 2)
            becnhmark_code.writeline("if __name__ == '__main__':")
            with becnhmark_code.indent():
                becnhmark_code.writeline(f"# Add your test code here")
                becnhmark_code.writeline(f"pass")
            f.write(becnhmark_code.getvalue())

    def load(self, name: str, index: sympy.Expr):
        buf: BufDesc = self._buf_desc[name]
        if not buf.src:
            self.graph.input(name, self.args.input(name))
            data = ir.data(name=name, sizes=buf.asc_size, dtype=buf.asc_dtype)
            buf.src = data
        else:
            data = buf.src

        if hasattr(self.current_node.node.data, 'input_sizes'):
            sizes = self.current_node.node.data.input_sizes[self._current_input_index]
        else:
            sizes = self.contiguous_loop.size
        self._current_input_index += 1
        loop = self._index_to_loop(index, sizes=sizes)
        scalars: Dict[str, _Scalar] = self._get_npu_scalar(index)
        if len(scalars):
            return ir.load_indirect(data.as_loop(loop), *[v.cse for v in scalars.values()], expr=str(index),
                                    syms=[f"{str(k)}={str(v.cse)}(\\<{v.max_value})" for k, v in scalars.items()])
        if loop.is_contiguous():
            load = ir.load(data.as_loop(loop=loop), loop=loop)
        else:
            road = self._get_view_road(loop, DenseLoop(axis=loop.axis, size=sizes))
            loop = road[0].src
            load = ir.load(data.as_loop(loop=loop), loop=loop)
            logging.info(f"Road for index {index} from {loop} to {self.contiguous_loop}")
            for op in road:
                logging.info(f"  {op.kind} from {op.src} to {op.dst}")
                load = getattr(ir, op.kind)(load, loop=op.dst)
        return load

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

    def index_to_str(self, index):
        return str(index)

    def _buf_size(self, buf):
        return [str(AscExpr(s)) for s in self._buf_desc[buf].size]

    def _buf_dtype(self, buf):
        return TypeUtils.torch_to_asc(self._buf_desc[buf].dtype)

    def _get_reduce_dims_and_loop(self, index: sympy.Expr):
        loop = self._index_to_loop(index)
        reduce_dims = [i for i in range(len(loop.stride)) if str(loop.stride[i]) == "0"]
        return reduce_dims, loop

    def _index_to_loop(self, index: sympy.Expr, axises=None, sizes=None):
        loop = Loop()
        loop.offset = index
        axises = axises if axises else self.contiguous_loop.axis
        sizes = sizes if sizes else self.contiguous_loop.size
        for axis, axis_size in zip(axises, sizes):
            loop.stride.append(index.coeff(axis))
            loop.offset = simplify(loop.offset.subs(axis, 0))
            loop.axis.append(axis)
            loop.size.append(sympy.S.One if str(loop.stride[-1]) == "0" else axis_size)
        return loop

    def _mark_buf_src(self, name, src):
        buf: BufDesc = self._buf_desc[name]
        if buf.is_output:
            data = ir.output(name=name, input=src, sizes=buf.asc_size, dtype=buf.asc_dtype)
        else:
            data = ir.workspace(name=name, input=src, sizes=buf.asc_size, dtype=buf.asc_dtype)
        buf.src = data
        if buf.is_output:
            self.graph.output(name, self.args.output(name))
        return data

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

    @classmethod
    def can_fuse_npu(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        def get_concat_nodes(node: BaseSchedulerNode):
            concats = []
            for n in node.get_nodes():
                if hasattr(n, 'node') and hasattr(n.node, 'data') and isinstance(n.node.data, UBConcat):
                    concats.append(n)
            return concats

        n1_concats = get_concat_nodes(node1)
        n2_concats = get_concat_nodes(node2)

        if len(n1_concats) > 1 or len(n2_concats) > 1:
            return False

        if len(n1_concats) == len(n2_concats):
            return False

        def is_user(src: BaseSchedulerNode, dst: BaseSchedulerNode):
            for node in src.get_nodes():
                if node.users is None:
                    continue
                if node.users and any([user.node == dst for user in node.users]):
                    return True
            return False

        return is_user(node2, n1_concats[0]) if n1_concats else is_user(node1, n2_concats[0])

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_vertical(node1, node2) or self.can_fuse_npu(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_horizontal(node1, node2)

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
        wrapper: WrapperCodeGen = V.graph.wrapper_code
        comments = _node_comment(nodes)
        for comment in comments:
            wrapper.writeline(comment)

        kernel = NPUKernel(nodes, comments=comments)
        for i, node in enumerate(nodes):
            logging.info(f"Codegen [{i+1}/{len(nodes)}] {node.debug_str()}")

            body = getattr(node, '_body')
            var_ranges = body.var_ranges
            indexing_exprs = body.indexing_exprs
            axis_indexings = []
            node_axis = dict()
            for axis, axis_size in dict(var_ranges).items():
                index = sympy.Symbol(f'{node.node.name}_{axis}') if len(nodes) > 1 else sympy.Symbol(axis.name)
                axis_indexings.append([index])
                node_axis[index] = V.graph.sizevars.simplify(axis_size)
                kernel.graph.axis(index.name, node_axis[index])

            for indexing_expr in itertools.chain(indexing_exprs, var_ranges.values()):
                indexing_expr = V.graph.sizevars.simplify(indexing_expr)
                size_vars = [s for s in indexing_expr.free_symbols if s.name.startswith('s')]
                for v in sorted(size_vars, key=lambda x: x.name):
                    kernel.graph.size(v.name)

            dense_loop = DenseLoop(axis=list(node_axis.keys()), size=list(node_axis.values()))
            with kernel, kernel.set_current_node(node), kernel.set_current_loop(dense_loop):
                node.run(*axis_indexings)

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

        from torch._inductor import config
        if config.trace.enabled:
            kernel.benchmark(V.debug.filename(f"{kernel.kernel_name}_benchmark.py"))
            kernel.view_dot(nodes, V.debug.filename(f"{kernel.graph.name}.svg"))
            kernel.record_summary(nodes, V.debug.filename(f"Model.csv"))

        _, call_args, _ = kernel.args.python_argdefs()

        # Manual combine size vars with tensor sizes
        workspace_var_name = f"{camel_to_snake(kernel.kernel_name)}_workspace"
        # Todo: symbolic workspace size
        device = f'{call_args[0]}.device' if len(call_args) else 'npu'
        wrapper.writeline("# Todo: symbolic workspace size")
        wrapper.writeline(f"{workspace_var_name} = torch.empty(1024 * 1024, device={device})")
        used_sizes = list(sorted(kernel.graph.size_vars))
        call_args.append(workspace_var_name)
        call_args.extend([f"{v}={v}" for v in used_sizes])
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
            """
        )
