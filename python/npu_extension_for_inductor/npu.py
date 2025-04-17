import dataclasses
import functools
import itertools
import contextlib
import logging
import os
from typing import List, Iterable, Dict, Union, Set
from unittest.mock import patch

from npu_extension_for_inductor.common.asc_graph import ASCGraph, FusedASCGraph
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
from torch._inductor.utils import get_kernel_metadata
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
        return ir.constant(repr(value), TypeUtils.torch_to_asc(dtype))

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

    @staticmethod
    def indirect_indexing(index_var, size, check=False) -> sympy.Symbol:
        kernel: NPUKernel = V.kernel
        return kernel.indirect_indexing(index_var, size, check)


class ASCBuffer:
    def __init__(self, name, layout):
        self.name = name
        self.dtype = layout.dtype
        self.size = [V.graph.sizevars.simplify(s) for s in layout.size]
        self.stride = [V.graph.sizevars.simplify(s) for s in layout.stride]
        self.offset = V.graph.sizevars.simplify(layout.offset)
        self.device = layout.device.type

        self._hint_size = [Loop.get_hint(s) for s in self.size]
        self._hint_stride = [Loop.get_hint(s) for s in self.stride]
        self._hint_offset = Loop.get_hint(self.offset)

    @property
    def asc_size(self):
        return [AscExpr(s) for s in self.size]

    @property
    def asc_dtype(self):
        return TypeUtils.torch_to_asc(self.dtype)

    def bind(self, src):
        src.op.set_private_attr(f'layout.device', self.device)
        src.op.set_private_attr(f'layout.dtype', self.dtype)
        src.op.set_private_attr(f'layout.size', self.size)
        src.op.set_private_attr(f'layout.stride', self.stride)
        src.op.set_private_attr(f'layout.offset', self.offset)

        src.op.set_private_attr(f'layout.hint.size', self._hint_size)
        src.op.set_private_attr(f'layout.hint.stride', self._hint_stride)
        src.op.set_private_attr(f'layout.hint.offset', self._hint_offset)
        return src


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


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    def __init__(self, nodes: List[BaseSchedulerNode], *, comments=None):
        super().__init__()
        self._comments: List[str] = comments
        self._kernel = NPUKernel.next_kernel_name()
        self.kernel_name = self._kernel
        self._kernel_def = IndentedBuffer()
        self._subgraphs: List[ASCGraph] = []
        self._indirect_to_scalar: Dict[str, _Scalar] = dict()
        self._current_loop = None
        self._current_input_index = 0
        self._asc_buffer: Dict[str:ASCBuffer] = {}
        self._torch_arg_wrappers = dict()
        self._nodes = nodes
        self._outputs = set()
        for node in nodes:
            user_nodes = set(user.node for user in node.users)
            if not user_nodes.issubset(nodes):
                self._outputs.add(node.node.name)

    @property
    def graph(self):
        assert len(self._subgraphs) > 0, "Graph is not initialized"
        return self._subgraphs[-1]

    @property
    def fused_graph(self):
        assert len(self._subgraphs) > 0, "Graph is not initialized"
        return FusedASCGraph(subgraphs=self._subgraphs, outputs=self._outputs, name=self._kernel)

    @property
    def contiguous_loop(self):
        return self._current_loop

    @property
    def assert_function(self):
        return "ascir.Assert"

    @staticmethod
    def _get_free_symbols(node: BaseSchedulerNode):
        body: LoopBody = getattr(node, '_body')
        free_symbols = set()
        for indexing_expr in itertools.chain(body.indexing_exprs, body.var_ranges.values()):
            indexing_expr = V.graph.sizevars.simplify(indexing_expr)
            size_vars = [s for s in indexing_expr.free_symbols if s.name.startswith('s')]
            for v in size_vars:
                free_symbols.add(v.name)
        return sorted(free_symbols)

    @staticmethod
    def _get_minimal_transpose_order(node: BaseSchedulerNode):
        body: LoopBody = getattr(node, '_body')
        min_score = None
        min_transpose_order = None
        for axis_vars in itertools.permutations(body.var_ranges.keys()):
            input_transposed = _get_transposed_indexing(body.reads_name2expr, axis_vars)
            for buffer, index in input_transposed:
                logging.debug("Reading index %s of %s is transposed under %s", index, buffer, axis_vars)
            output_transposed = _get_transposed_indexing(body.writes_name2expr, axis_vars)
            for buffer, index in output_transposed:
                logging.debug("Writing index %s of %s is transposed under %s", index, buffer, axis_vars)
            score = len(input_transposed) + len(output_transposed)
            logging.debug("Totally %s transposed indexings under %s", score, axis_vars)
            if min_score is None or score < min_score:
                min_score = score
                min_transpose_order = axis_vars
            if min_score == 0:
                break
        logging.debug("Finally transposed order is %s with score %s", min_transpose_order, min_score)
        return min_transpose_order

    @classmethod
    def next_kernel_name(cls):
        name = f"fused_asc_{cls._index}"
        cls._index += 1
        return name

    @contextlib.contextmanager
    def new_subgraph(self, free_symbols: Set[str], asc_axis: List[sympy.Symbol], asc_axis_range: List[sympy.Expr]):
        loop = DenseLoop(axis=asc_axis, size=asc_axis_range)
        self._subgraphs.append(ASCGraph(name=f"{V.kernel.current_node.node.name}_asc"))
        self._current_input_index = 0
        self.graph.set_current_loop(loop)
        for axis, axis_range in zip(asc_axis, asc_axis_range):
            self.graph.axis(axis.name, axis_range)
        for s in free_symbols:
            self.graph.size(s)
        prior = self._current_loop
        self._current_loop = loop
        try:
            yield
        finally:
            self._current_loop = prior

    def tracing_asc(self):
        with self:
            for i, node in enumerate(self._nodes):
                logging.info("Codegen [%s] %s", f"{i+1}/{len(self._nodes)}", node.debug_str())
                body: LoopBody = getattr(node, '_body')

                free_symbols = self._get_free_symbols(node)

                var_to_asc_axis = {}
                axis_indexings = []
                for var in body.var_ranges.keys():
                    var_to_asc_axis[var] = sympy.Symbol(var.name)
                    axis_indexings.append([var_to_asc_axis[var]])

                asc_axis = []
                asc_axis_range = []
                for var in self._get_minimal_transpose_order(node):
                    asc_axis.append(var_to_asc_axis[var])
                    asc_axis_range.append(V.graph.sizevars.simplify(body.var_ranges[var]))

                with self.set_current_node(node), self.new_subgraph(sorted(free_symbols), asc_axis, asc_axis_range):
                    node.run(*axis_indexings)
                    logging.info(f"{self.graph.name} reads {self.graph.inputs} and writes {self.graph.outputs}")

        V.graph.removed_buffers |= self.removed_buffers
        if hasattr(self, 'inplaced_to_remove') and hasattr(V.graph, 'inplaced_to_remove'):
            V.graph.inplaced_to_remove |= self.inplaced_to_remove

        return self

    def get_asc_buffer(self, name):
        if name in self._asc_buffer:
            return self._asc_buffer[name]
        buf = V.graph.get_buffer(name)
        layout_size = [V.graph.sizevars.simplify(s) for s in buf.layout.size]
        for size_expr in layout_size:
            size_vars = [s for s in size_expr.free_symbols if s.name.startswith('s')]
            for v in size_vars:
                self.graph.size(v.name)

        self._asc_buffer[name] = ASCBuffer(name, buf.layout)
        return self._asc_buffer[name]

    def codegen(self):
        fused_graph = self.fused_graph
        # 对于输出复用输入的场景，可能出现多个asc graph上的buffer（Data/Output）对应同一个python kernel入参的情况，
        # outer是python kernel层的入参名，而inputs/outputs，则是asc graph上的buffer名，也对应rt层kernel的args
        fused_graph.inputs_outer = [self.args.input(read) for read in fused_graph.inputs]
        fused_graph.outputs_outer = [self.args.output(write) for write in fused_graph.outputs]
        # 这里的args，对应python kernel的入参，而第二个返回，是在output code call函数中，调用python kernel时传入的参数
        fused_graph.args, _, _ = self.args.python_argdefs()
        used_symbols = [str(v) for v in fused_graph.size_vars]

        signature = ', '.join(fused_graph.args + ['workspace'] + used_symbols)
        call_args = ', '.join(fused_graph.args + ['workspace'] + [f"{v}={v}" for v in used_symbols])

        self._kernel_def.clear()
        kernel_var_name = f"{self._kernel}_compiled"

        from npu_extension_for_inductor import codegen as npu_codegen
        self._kernel_def.splice(npu_codegen.codegen_kernel_def(fused_graph, kernel_var_name))
        self._kernel_def.writelines(self._comments)
        self._kernel_def.writeline(f"def {self._kernel}({signature}):")
        with self._kernel_def.indent():
            for k, v in self._torch_arg_wrappers.items():
                self._kernel_def.writeline(f"{k} = {v}")
            self._kernel_def.writeline(f"{kernel_var_name}({call_args})")

        return self._kernel_def.getvalue()

    def record_summary(self, nodes, model_path=None):
        for i, graph in enumerate(self._subgraphs):
            loop_body = _node_label(nodes[i]) if i < len(nodes) else ""
            OP_SUMMARY.add_graph_summary(graph, loop=loop_body, model_path=model_path)

    def view_dot(self, nodes, svg_path=None):
        try:
            import pydot
            dot_graph = self.fused_graph.as_dot()
            labels = [_node_label(node) + ['-' * 20] for node in nodes]
            lines = list(itertools.chain(*labels))
            lines = _left_align_lines(lines)
            dot_graph.add_node(
                pydot.Node(f"{self.kernel_name}_body", shape="plaintext", label='\n'.join(lines),
                           fontname="Courier"))
            svg_path = svg_path if svg_path else f"./{self.kernel_name}.svg"
            dot_graph.write_svg(svg_path)
        except ImportError:
            logging.info("Unable to save dot for kernel %s as pydot not installed", self.kernel_name)

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
        ir_data = self.current_node.node.data
        if isinstance(ir_data, UBConcat):
            sizes = self.contiguous_loop.size[:]
            concat_size = ir_data.output_concat_dim_size
            for i, size in enumerate(self.contiguous_loop.size):
                coeff = size.coeff(concat_size)
                if str(coeff) != '0':
                    sizes[i] = coeff * ir_data.input_concat_dim_sizes[self._current_input_index]
        else:
            sizes = self.contiguous_loop.size
        self._current_input_index += 1
        scalars: Dict[str, _Scalar] = self._get_npu_scalar(index)
        if len(scalars):
            return ir.load_indirect(self._load_indirect_buffer(name), *[v.cse for v in scalars.values()],
                                    expr=str(index),
                                    syms=[f"{str(k)}={str(v.cse)}(\\<{v.max_value})" for k, v in scalars.items()])

        data, loop = self._load_buffer(name, self._index_to_loop(index, sizes=sizes))
        offset = loop.zero_offset_()
        road = self._get_view_road(loop, DenseLoop(axis=loop.axis, size=sizes))
        if len(road) == 0:
            logging.info("Road for %s from %s to %s is dense", index, loop, self.contiguous_loop)
            return ir.load(data, offset=offset, loop=loop)
        loop = road[0].src
        load = ir.load(data, offset=offset, loop=loop)
        logging.info("Road for %s from %s to %s", index, loop, self.contiguous_loop)
        for op in road:
            logging.info("  %s from %s to %s", op.kind, op.src, op.dst)
            load = getattr(ir, op.kind)(load, loop=op.dst)
        return load

    def store_reduction(self, name, index, value: Reduction):
        reduce_dims, loop = self._get_reduce_dims_and_loop(index)
        reduction = ir.reduction(value.value, reduce_type=value.reduction_type, loop=loop)
        reduction = NPUOverrides.to_dtype(reduction, dst_dtype=value.dtype, src_dtype=value.src_dtype)

        store = ir.store(reduction, loop=loop)
        value.src = self._store_buffer(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def store(self, name, index, value, mode=None):
        store = ir.store(value, loop=self._index_to_loop(index))
        self._store_buffer(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def reduction(self, dtype, src_dtype, reduction_type, value):
        return Reduction(dtype, src_dtype, reduction_type, value, '')

    def rename_indexing(self, index) -> sympy.Expr:
        if isinstance(index, sympy.Symbol) and index.name.startswith("s"):
            self.graph.size(index.name)
            return index
        return super().rename_indexing(index)

    def indirect_indexing(self, index_var, size, check=False) -> sympy.Symbol:
        indirect_sym = sympy.Symbol(f"npu_scalar{len(self._indirect_to_scalar)}")
        op_name, output_name = str(index_var).split('.')
        src = self.graph.get_op(op_name)
        assert src is not None
        self._indirect_to_scalar[str(indirect_sym)] = _Scalar(_Tensor(getattr(src, output_name)), size, check)
        return indirect_sym

    def index_to_str(self, index):
        return str(index)

    def _load_indirect_buffer(self, name):
        buf: ASCBuffer = self.get_asc_buffer(name)
        exist_tensor = self.graph.get_tensor(name)
        if exist_tensor is not None:
            return exist_tensor
        return buf.bind(self.graph.input(name, buf.asc_dtype))

    def _load_buffer(self, name, loop: Loop):
        buf: ASCBuffer = self.get_asc_buffer(name)
        exist_tensor = self.graph.get_tensor(name)
        if exist_tensor is not None:
            return exist_tensor, loop
        return buf.bind(self.graph.input(name, buf.asc_dtype)), loop

    def _store_buffer(self, name, src):
        buf: ASCBuffer = self.get_asc_buffer(name)
        return buf.bind(self.graph.output(name, buf.asc_dtype, src=src))

    def _get_reduce_dims_and_loop(self, index: sympy.Expr):
        loop = self._index_to_loop(index)
        reduce_dims = [i for i in range(len(loop.stride)) if str(loop.stride[i]) == "0"]
        return reduce_dims, loop

    def _index_to_loop(self, index: sympy.Expr, axises=None, sizes=None):
        loop = Loop()
        loop.offset = index
        axises = axises if axises else self.contiguous_loop.axis
        sizes = sizes if sizes else self.contiguous_loop.size

        loop.stride = V.graph.sizevars.stride_vars(index, axises)
        loop.offset = V.graph.sizevars.offset_var(index, axises)
        loop.axis = axises
        loop.size = [sympy.S.One if str(loop.stride[i]) == "0" else s for i, s in enumerate(sizes)]

        return loop

    def _get_npu_scalar(self, index: sympy.Expr):
        scalars = dict()
        for s in index.free_symbols:
            if str(s) in self._indirect_to_scalar:
                scalars[s] = self._indirect_to_scalar[str(s)]
        return scalars

    def _get_view_road(self, src: Loop, dst: DenseLoop):
        if src == dst:
            return []
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
        src_loop = src.copy()
        for i, j in zip(range(len(order)), order):
            if i != j:
                road_dst = road[0].src if road else dst
                road_src = road_dst.copy().transpose_(i, j).contiguous_()
                src_loop.transpose_(i, j)
                road.insert(0, MoveOp(kind="transpose", src=road_src, dst=road_dst))
                order[i], order[j] = order[j], order[i]

        road_dst = road[0].src if road else dst
        broadcast_dims = [i for i, (src_size, dst_size) in enumerate(zip(src_loop.size, road_dst.size))
                          if str(src_size) == '1' and str(src_size) != str(dst_size)]
        for dim in broadcast_dims:
            road_dst = road[0].src if road else dst
            road.insert(0, MoveOp(kind="broadcast", src=road_dst.copy().debroadcast_(dim), dst=road_dst))

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


def _get_transposed_indexing(load_index, axis_vars):
    transposed_index = []
    for buffer, index in load_index.items():
        hints = V.graph.sizevars.stride_hints(index, axis_vars)
        non_zero_hints = [hint for hint in hints if str(hint) != '0']
        if sorted(non_zero_hints, reverse=True) != non_zero_hints:
            transposed_index.append((buffer, index))
    return transposed_index


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self._fuse_judge = TritonScheduling(scheduler)

    @classmethod
    def can_fuse_npu(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if not all([isinstance(node, (SchedulerNode, FusedSchedulerNode)) for node in [node1, node2]]):
            return False

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

        kernel = NPUKernel(nodes, comments=comments).tracing_asc()

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

        from torch._inductor import config
        if config.trace.enabled:
            kernel.benchmark(V.debug.filename(f"{kernel.kernel_name}/benchmark.py"))
            kernel.view_dot(nodes, V.debug.filename(f"{kernel.kernel_name}/graph.svg"))
            kernel.record_summary(nodes, V.debug.filename(f"{kernel.kernel_name}/fuse_summary.csv"))

        _, call_args, _ = kernel.args.python_argdefs()

        # Manual combine size vars with tensor sizes
        workspace_var_name = f"{camel_to_snake(kernel.kernel_name)}_workspace"
        # Todo: symbolic workspace size
        device = f'{call_args[0]}.device' if len(call_args) else "'npu'"
        wrapper.writeline("# Todo: symbolic workspace size")
        wrapper.writeline(f"{workspace_var_name} = torch.empty(1024 * 1024 * 1024 // 4, device={device})")
        used_sizes = list(kernel.fused_graph.size_vars)
        call_args.append(workspace_var_name)
        call_args.extend([f"{v}={v}" for v in used_sizes])
        if os.getenv("NPU_INDUCTOR_DEBUG_SINGLE_KERNEL", None) == '1':
            for arg in [str(v) for v in call_args][:len(call_args) - len(used_sizes)]:
                wrapper.writeline(f"print({repr(kernel.kernel_name)}, {repr(arg)}, {arg}.device, {arg}.dtype, "
                                  f"{arg}.shape, {arg}.stride(), {arg}.storage_offset(), flush=True)")
        wrapper.writeline(wrapper.wrap_kernel_call(kernel.kernel_name, [str(v) for v in call_args]))
        wrapper.writeline(f"del {workspace_var_name}")
        if os.getenv("NPU_INDUCTOR_DEBUG_SINGLE_KERNEL", None) == '1':
            wrapper.writeline(f"print('Start synchronize kernel {kernel.kernel_name}', flush=True)")
            wrapper.writeline(f"torch.npu.synchronize()")
            wrapper.writeline(f"print('Finish synchronize kernel {kernel.kernel_name}', flush=True)")

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
