# pylint: disable=W1203,E1125,W1514,R1729,W0246,W0201
import dataclasses
import itertools
import contextlib
import hashlib
import os
from typing import List, Dict, Union, Set
from collections import OrderedDict

import sympy
import torch

from sympy.printing.pycode import pycode
from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.ir import LoopBody
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode, FusedSchedulerNode
from torch._inductor.utils import get_kernel_metadata, get_fused_kernel_name
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import IndentedBuffer, Kernel, TensorArg

from .common import logger
from .common import fused_layout_check
from .common.asc_graph import ASCGraph, FusedASCGraph, ASCIndexing
from .common.symbols import Axis
from .common.debug import _left_align_lines, OP_SUMMARY
from .common.symbols import AscExpr, Loop, DenseLoop
from .common.asc_graph import _Tensor
from . import asc_ops as ir
from .asc_overrides import NPUOverrides
from .config import _debug_options


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

    def bind(self, src: _Tensor) -> _Tensor:
        src.op.set_private_attr('layout.device', self.device)
        src.op.set_private_attr('layout.dtype', self.dtype)
        src.op.set_private_attr('layout.size', self.size)
        src.op.set_private_attr('layout.stride', self.stride)
        src.op.set_private_attr('layout.offset', self.offset)

        src.op.set_private_attr('layout.hint.size', self._hint_size)
        src.op.set_private_attr('layout.hint.stride', self._hint_stride)
        src.op.set_private_attr('layout.hint.offset', self._hint_offset)
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


def _get_nodes_outputs(nodes: List[BaseSchedulerNode]):
    from torch._inductor.scheduler import OutputNode

    buffers = []
    for node in nodes:
        for output in node.outputs:
            for user in output.users:
                if isinstance(user.node, OutputNode) or user.node not in nodes:
                    buffers.append(output.node.name)
                    break
    return list(OrderedDict.fromkeys(buffers))


class NPUKernel(Kernel):
    overrides = NPUOverrides
    _index = 0

    class Artifacts:
        def __init__(self, *, name, tiling_def, host_impl, device_impl, cpp_wrapper):
            self.name = name
            self.tiling_def = tiling_def
            self.host_impl = host_impl
            self.device_impl = device_impl
            self.cpp_wrapper = cpp_wrapper

    def __init__(self, nodes: List[BaseSchedulerNode], *, comments=None):
        super().__init__()
        self._comments: List[str] = comments
        self._artifacts: NPUKernel.Artifacts = None
        self._graph: ASCGraph = None  # 单图：所有节点共用一个 ASCGraph
        self._current_loop = None
        self._asc_buffer: Dict[str, ASCBuffer] = {}
        self._torch_arg_wrappers = dict()
        self._nodes = nodes
        self._outputs = _get_nodes_outputs(nodes)
        # name -> (value tensor, loop)，记录 fused 内部已 store 的 buffer，下次 load 直接复用，避免在同图中
        # 同时出现 Data+Output 形成 FusedGraph 自环。
        self._local_stores: Dict[str, tuple] = {}

    @property
    def graph(self):
        return self._graph

    @property
    def fused_graph(self):
        return self._fused_graph

    @property
    def contiguous_loop(self):
        return self._current_loop

    @property
    def assert_function(self):
        return "ascir.Assert"

    def size_hint(self, expr: Union[sympy.Expr, int]):
        if isinstance(expr, sympy.Expr):
            replacements = {}
            for s, ks in self.args.sizevars.items():
                for sym in expr.free_symbols:
                    if sym.name == ks:
                        replacements[sym] = s
            expr = sympy_subs(expr, replacements)
        return V.graph.sizevars.size_hint(expr, fallback=-1)

    @staticmethod
    def _get_minimal_transpose_order(node: BaseSchedulerNode):
        body: LoopBody = getattr(node, '_body')
        min_score = None
        min_transpose_order = None
        for axis_vars in itertools.permutations(body.var_ranges.keys()):  # dict[axis:range]循环轴和对应的大小
            index_transposed = _get_transposed_indexing(
                body.indexing_exprs, axis_vars
            )  # dict[idx:expr]读写内存使用的index
            for idx_name, expr, score in index_transposed:
                logger.debug("Expr %s of %s is transposed score %s under %s", expr, idx_name, score, axis_vars)
            score = sum(score for _, _, score in index_transposed)
            logger.debug("Totally transposed indexings score %s under %s", score, axis_vars)
            if min_score is None or score < min_score:
                min_score = score
                min_transpose_order = axis_vars
            if min_score == 0:
                break
        logger.debug("Finally transposed order is %s with score %s", min_transpose_order, min_score)
        return min_transpose_order

    def _canonical_axes_for_kernel(self):
        """选择 axis 数量最多（最细分）的节点作为 canonical，统一改名为 a0/a1/...
        以避免和其他节点 var 同名冲突。其他节点会通过 contiguous flatten 多项式
        映射到这组 canonical 轴。"""
        chosen_sizes: List[sympy.Expr] = []
        for node in self._nodes:
            body: LoopBody = getattr(node, '_body')
            order = self._get_minimal_transpose_order(node)
            sizes = [self.rename_indexing(body.var_ranges[v]) for v in order]
            if len(sizes) > len(chosen_sizes):
                chosen_sizes = sizes
        if not chosen_sizes:
            return ["a0"], [sympy.S.One]
        names = [f"a{i}" for i in range(len(chosen_sizes))]
        return names, chosen_sizes

    @staticmethod
    def _flatten_expr(group_axes, group_sizes):
        """对一组连续 canonical 轴，按 contiguous 顺序生成 flatten 多项式：
        a0 * (s1 * s2 * ...) + a1 * (s2 * ...) + ... + a_{n-1}"""
        expr = sympy.S.Zero
        for j, axis in enumerate(group_axes):
            inner_prod = sympy.S.One
            for k in range(j + 1, len(group_sizes)):
                inner_prod = inner_prod * group_sizes[k]
            expr = expr + axis * inner_prod
        return expr

    def _node_axis_indexings(self, node, canonical_axes, canonical_ranges):
        """计算 node.run() 的 axis_indexings：把每个 node 轴映射成一组连续 canonical
        轴的 flatten 多项式。允许节点比 canonical 轴少（pointwise collapse 场景），
        通过把单个 node 轴展开成多个 canonical 轴的线性组合来对齐。
        如果 node 的总迭代量比 canonical 还少（典型场景：reduce 之后接 pointwise，
        被 scheduler 融到了 reduce kernel），剩余 canonical 轴对该 node 等价于 broadcast，
        load/store 在这些轴上 stride=0，AscIR 由 size 推导出实际 tile。"""
        body: LoopBody = getattr(node, '_body')
        transpose_order = self._get_minimal_transpose_order(node)
        node_sizes = [self.rename_indexing(body.var_ranges[v]) for v in transpose_order]

        var_to_expr: Dict[sympy.Symbol, sympy.Expr] = {}
        canonical_idx = 0
        for node_var, node_size in zip(transpose_order, node_sizes):
            group_axes: List[sympy.Symbol] = []
            group_sizes: List[sympy.Expr] = []
            product = sympy.S.One
            while canonical_idx < len(canonical_axes):
                group_axes.append(canonical_axes[canonical_idx])
                group_sizes.append(canonical_ranges[canonical_idx])
                product = product * canonical_ranges[canonical_idx]
                canonical_idx += 1
                if sympy.simplify(product - node_size) == 0:
                    break
            if sympy.simplify(product - node_size) != 0:
                raise RuntimeError(
                    f"Cannot map node axis {node_var}(size={node_size}) into canonical axes "
                    f"with sizes {canonical_ranges}")
            var_to_expr[node_var] = self._flatten_expr(group_axes, group_sizes)

        axis_indexings: List[List[sympy.Expr]] = []
        for var in body.var_ranges.keys():
            expr = var_to_expr.get(var)
            if expr is None:
                expr = sympy.Symbol(var.name)
            axis_indexings.append([expr])
        return axis_indexings

    @staticmethod
    def _node_loop_sizes(axis_indexings, canonical_axes, canonical_ranges):
        """根据 node 实际用到的 canonical 轴推导 per-node contiguous_loop 的 size。
        没有被 axis_indexings 引用的 canonical 轴，对该 node 等价于广播——size 设为 1，
        DenseLoop 会自动给出 stride=0。这样像 to_dtype 这类没有显式传 loop 的算子，
        默认拿到的形状就和后续基于 index 推导出的 store loop 一致，不会出现 Cast 输入/输出
        size 不匹配的非法图。"""
        used: Set[str] = set()
        for expr_list in axis_indexings:
            for expr in expr_list:
                if isinstance(expr, sympy.Expr):
                    for sym in expr.free_symbols:
                        used.add(sym.name)
        return [canonical_ranges[i] if axis.name in used else sympy.S.One
                for i, axis in enumerate(canonical_axes)]

    def tracing_asc(self):
        with self:
            canonical_names, canonical_ranges = self._canonical_axes_for_kernel()
            canonical_axes = [sympy.Symbol(n) for n in canonical_names]
            canonical_loop = DenseLoop(axis=canonical_axes, size=canonical_ranges)

            hint_lines = []
            for node in self._nodes:
                hint_lines.extend(_node_label(node))
                hint_lines.append('-' * 20)
            hint_str = '\n'.join(hint_lines).rstrip('-\n')

            self._graph = ASCGraph(name="graph", hint_str=hint_str)
            for axis, axis_range in zip(canonical_axes, canonical_ranges):
                self._graph.axis(axis.name, axis_range)
            self._graph.set_current_loop(canonical_loop)

            prior_loop = self._current_loop
            try:
                for i, node in enumerate(self._nodes):
                    logger.debug("Codegen [%s] %s", f"{i+1}/{len(self._nodes)}", node.debug_str())
                    axis_indexings = self._node_axis_indexings(node, canonical_axes, canonical_ranges)
                    node_sizes = self._node_loop_sizes(axis_indexings, canonical_axes, canonical_ranges)
                    self._current_loop = DenseLoop(axis=canonical_axes, size=node_sizes)
                    with self.set_current_node(node):
                        node.run(*axis_indexings)
                    logger.debug(f"{self._graph.name} reads {self._graph.inputs} and writes {self._graph.outputs}")
            finally:
                self._current_loop = prior_loop

        if hasattr(self, 'removed_buffers') and hasattr(V.graph, 'removed_buffers'):
            V.graph.removed_buffers |= self.removed_buffers
        if hasattr(self, 'inplaced_to_remove') and hasattr(V.graph, 'inplaced_to_remove'):
            V.graph.inplaced_to_remove |= self.inplaced_to_remove

        for sym, sym_renamed in self.args.sizevars.items():
            self._graph.size(sym_renamed)

        self._fused_graph = FusedASCGraph(graph=self._graph, outputs=self._outputs)
        # 对于输出复用输入的场景，可能出现多个asc graph上的buffer（Data/Output）对应同一个python kernel入参的情况，
        # outer是python kernel层的入参名，而inputs/outputs，则是asc graph上的buffer名，也对应rt层kernel的args
        self._fused_graph.inputs_outer = [self.args.input(read) for read in self._fused_graph.inputs]
        self._fused_graph.outputs_outer = [self.args.output(write) for write in self._fused_graph.outputs]
        # 这里的args，对应python kernel签名的入参名字，也是wrapper签名中的入参名字。
        # 而第二个返回，是在output code call函数中，调用python kernel时传入的参数，也就是实际buffer的名字。
        arg_defs, call_args, precompile_args, arg_types = self.args.python_argdefs()
        self._fused_graph.args = precompile_args

        from . import codegen as npu_codegen

        self._fused_graph.cpp_wrapper = npu_codegen.codegen_cpp_wrapper(self._fused_graph)
        self._fused_graph.asc_graph = self._fused_graph.codegen("cache_hint").getvalue()
        md5 = hashlib.md5(f"{self._fused_graph.asc_graph}_{self._fused_graph.cpp_wrapper}".encode()).hexdigest()  # nosec B324
        self._fused_graph.name = f"auto{get_fused_kernel_name(self._nodes, 'original_aten')}_{md5}"

        unsupported_ops = set(self._graph.unsupported_ops)

        if unsupported_ops:
            self._fused_graph.name = f"unsupported_{'_'.join(sorted(unsupported_ops))}_{self._fused_graph.name}"

        return self

    def get_asc_buffer(self, name):
        if name in self._asc_buffer:
            return self._asc_buffer[name]
        buf = V.graph.get_buffer(name)
        self._asc_buffer[name] = ASCBuffer(name, buf.layout)
        return self._asc_buffer[name]

    def codegen(self):
        from . import codegen as npu_codegen

        artifacts = npu_codegen.codegen_kernel_def(self.fused_graph)
        artifacts['cpp_wrapper'] = npu_codegen.codegen_cpp_wrapper(self.fused_graph)
        if not all(v.strip() for v in artifacts.values()):
            raise RuntimeError(f"Failed to generate artifacts for kernel {self.kernel_name}: {artifacts}")

        self._artifacts = NPUKernel.Artifacts(**artifacts)  # noqa

        kernel_def = IndentedBuffer()
        kernel_obj = f"{self._artifacts.name}_artifacts"
        kernel_def.writeline(f"{kernel_obj} = {{}}")
        kernel_def.splice(f"{kernel_obj}['name'] = r'''{self._artifacts.name}'''")
        kernel_def.splice(f"{kernel_obj}['tiling_def'] = r'''{self._artifacts.tiling_def}'''")
        kernel_def.splice(f"{kernel_obj}['host_impl'] = r'''{self._artifacts.host_impl}'''")
        kernel_def.splice(f"{kernel_obj}['device_impl'] = r'''{self._artifacts.device_impl}'''")
        kernel_def.splice(f"{kernel_obj}['cpp_wrapper'] = r'''{self._artifacts.cpp_wrapper}'''")
        kernel_def.writeline(
            f"{self.kernel_name} = async_compile_ascendc(globals().get('async_compile', None), {kernel_obj})"
        )

        return kernel_def.getvalue()

    def record_summary(self, nodes, model_path=None):
        loop_body_lines = []
        for node in nodes:
            loop_body_lines.extend(_node_label(node))
            loop_body_lines.append('-' * 20)
        OP_SUMMARY.add_graph_summary(self._graph, loop='\n'.join(loop_body_lines).rstrip('-\n'),
                                     model_path=model_path)

    def view_dot(self, nodes, svg_path=None):
        try:
            import pydot

            dot_graph = self.fused_graph.as_dot()
            symbol_to_hint = []
            for s, ks in self.args.sizevars.items():
                symbol_to_hint.append(f'{s.name}:(={ks}, hint={self.size_hint(s)})')
            labels = [_node_label(node) + ['-' * 20] for node in nodes]
            lines = list(itertools.chain(symbol_to_hint, ['-' * 20], *labels))
            lines = _left_align_lines(lines)
            dot_graph.add_node(
                pydot.Node(f"{self.kernel_name}_body", shape="plaintext", label='\n'.join(lines), fontname="Courier")
            )
            svg_path = svg_path if svg_path else f"./{self.kernel_name}.svg"
            dot_graph.write_svg(svg_path)
        except ImportError:
            logger.warning("Unable to save dot for kernel %s as pydot not installed", self.kernel_name)
        except AssertionError:
            logger.warning("Unable to save dot for kernel %s as graphviz inner error", self.kernel_name)

    def benchmark(self, nodes, file_path=None):
        file_path = file_path if file_path else f"./{self.kernel_name}_benchmark.py"

        arg_defs, call_args, precompile_args, arg_types = self.args.python_argdefs()
        used_buffers = []
        seen_symbols = []
        for buffer, buffer_type in zip(call_args, precompile_args):
            if not isinstance(buffer_type, TensorArg):
                continue
            used_buffers.append(buffer)
            layout = V.graph.get_buffer(buffer).layout
            for expr in itertools.chain(layout.stride or [], layout.size or [], [layout.offset]):
                seen_symbols.extend(
                    V.graph.sizevars.simplify(expr).free_symbols if isinstance(expr, sympy.Expr) else []
                )

        with open(file_path, "w") as f:  # noqa
            benchmark_code = IndentedBuffer()
            benchmark_code.writeline("import sys")
            benchmark_code.writeline("import torch")
            benchmark_code.writeline("import torch_npu")
            benchmark_code.writeline(f"from {__package__}.compiler import async_compile as async_compile_ascendc")
            kernel_obj = f"{self._artifacts.name}_artifacts"
            benchmark_code.writeline(f"{kernel_obj} = {{}}")
            benchmark_code.splice(f"{kernel_obj}['name'] = r'''{self._artifacts.name}'''")
            benchmark_code.splice(f"{kernel_obj}['cpp_wrapper'] = r'''{self._artifacts.cpp_wrapper}'''")

            benchmark_code.writelines(["\n"] * 2)
            benchmark_code.writeline("if __name__ == '__main__':")
            with benchmark_code.indent():
                benchmark_code.writeline(
                    f"assert len(sys.argv) == 1 or sys.argv[-1] == 'e2e', 'Usage: python {file_path} [e2e]'"
                )
                benchmark_code.writeline("if sys.argv[-1] == 'e2e':")
                with benchmark_code.indent():
                    with open(os.path.join(os.path.dirname(file_path), "asc_graph.py"), "r") as asc_graph:  # noqa
                        benchmark_code.splice(asc_graph.read())
                    benchmark_code.splice(f"{kernel_obj}['tiling_def'] = tiling_def")
                    benchmark_code.splice(f"{kernel_obj}['host_impl'] = host_impl")
                    benchmark_code.splice(f"{kernel_obj}['device_impl'] = device_impl")
                benchmark_code.writeline("else:")
                with benchmark_code.indent():
                    benchmark_code.splice(f"{kernel_obj}['tiling_def'] = r'''{self._artifacts.tiling_def}'''")
                    benchmark_code.splice(f"{kernel_obj}['host_impl'] = r'''{self._artifacts.host_impl}'''")
                    benchmark_code.splice(f"{kernel_obj}['device_impl'] = r'''{self._artifacts.device_impl}'''")

                benchmark_code.writeline(f"{self.kernel_name} = async_compile_ascendc(None, {kernel_obj})")
                benchmark_code.writeline("from torch._dynamo.testing import rand_strided")
                for s, ks in self.args.sizevars.items():
                    benchmark_code.writeline(f"{ks} = {s} = {self.size_hint(s)}")
                for k in seen_symbols:
                    if k not in self.args.sizevars.keys():
                        benchmark_code.writeline(f"{k} = {self.size_hint(k)} # buffer size hint")
                for buffer in used_buffers:
                    layout = V.graph.get_buffer(buffer).layout
                    benchmark_code.writeline(
                        f"{buffer} = rand_strided({tuple(layout.size)}, {tuple(layout.stride)}, "
                        f"device='{layout.device if layout.device.type != 'npu' else 'npu'}', dtype={layout.dtype})"
                    )

                benchmark_code.splice("""
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
                        export_type=[
                            torch_npu.profiler.ExportType.Text,
                            torch_npu.profiler.ExportType.Db
                            ],
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None
                    )

                    with torch_npu.profiler.profile(
                        activities=[
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU
                            ],
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=10, repeat=1, skip_first=1),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                        with_modules=False,
                        with_flops=False,
                        experimental_config=experimental_config) as prof:
                """)
                with benchmark_code.indent():
                    benchmark_code.splice("for _ in range(11):")
                    with benchmark_code.indent():
                        benchmark_code.writeline(f"{self.kernel_name}({', '.join([str(v) for v in call_args])})")
                        benchmark_code.writeline("prof.step()")
            f.write(benchmark_code.getvalue())

    def load(self, name: str, index: sympy.Expr):
        if any([isinstance(s, ASCIndexing) for s in index.free_symbols]):  # noqa
            return self.indirect_load(name, index)

        index = self.rename_indexing(index)
        sizes = self.contiguous_loop.size

        # 本地融合：当前 fused kernel 内已经 store 过 name，直接复用值，避免在同一图中
        # 同时存在 Data 和 Output 形成 FusedGraph 自环。
        if name in self._local_stores:
            value, src_loop = self._local_stores[name]
            return self._reshape_local_value(value, src_loop, sizes)

        data, loop = self._load_buffer(name, self._index_to_loop(index, sizes=sizes))
        offset = loop.zero_offset_()
        road = self._get_view_road(loop, DenseLoop(axis=loop.axis, size=sizes))

        dtype = self.get_asc_buffer(name).dtype
        if len(road) == 0:
            logger.debug("Road for %s from %s to %s is dense", index, loop, self.contiguous_loop)
            load = ir.load(data, offset=offset, loop=loop)
            if dtype in {torch.bfloat16, torch.float16}:
                load = ir.cast(load, dst=torch.float32, loop=loop.copy().contiguous_())
            return load

        loop = road[0].src
        load = ir.load(data, offset=offset, loop=loop)
        if dtype in {torch.bfloat16, torch.float16}:
            load = ir.cast(load, dst=torch.float32, loop=loop.copy().contiguous_())

        logger.debug("Road for %s from %s to %s", index, loop, self.contiguous_loop)
        for op in road:
            logger.debug("  %s from %s to %s", op.kind, op.src, op.dst)
            load = getattr(ir, op.kind)(load, loop=op.dst)
        return load

    def _reshape_local_value(self, value: _Tensor, src_loop: Loop, dst_sizes):
        """把上游 store 的 value 调整到当前请求的形状（必要时插入 broadcast/transpose）。"""
        dst_loop = DenseLoop(axis=src_loop.axis, size=dst_sizes)
        road = self._get_view_road(src_loop.copy(), dst_loop)
        if not road:
            return value
        result = value
        for op in road:
            result = getattr(ir, op.kind)(result, loop=op.dst)
        return result

    def store(self, name, index, value, mode=None):
        index = self.rename_indexing(index)
        dtype = self.get_asc_buffer(name).dtype
        loop = self._index_to_loop(index)
        if dtype in {torch.bfloat16, torch.float16}:
            value = ir.cast(value, dst=torch.float32, loop=loop)
            value = ir.cast(value, dst=dtype, loop=loop)
        result = self._store_buffer(name, value, loop)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return result

    def reduction(self, dtype, src_dtype, reduction_type, value):
        reduction = ir.reduction(value, reduce_type=reduction_type)
        reduction.dtype = dtype
        return reduction

    def store_reduction(self, name, index, reduction: _Tensor):
        index = self.rename_indexing(index)
        reduce_dims, loop = self._get_reduce_dims_and_loop(index)
        reduction.as_loop(loop)
        dtype = self.get_asc_buffer(name).dtype
        if dtype in {torch.bfloat16, torch.float16}:
            reduction = ir.cast(reduction, dst=torch.float32, loop=loop)
            reduction = ir.cast(reduction, dst=dtype, loop=loop)
        result = self._store_buffer(name, reduction, loop)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return result

    def rename_indexing(self, index) -> sympy.Expr:
        return super().rename_indexing(index)

    def indirect_load(self, name: str, index: sympy.Expr) -> _Tensor:
        data, loop = self._load_buffer(name, self.contiguous_loop)
        asc_tensors = [s.asc_tensor for s in index.free_symbols if isinstance(s, ASCIndexing)]
        load = ir.indirect_load(data, *asc_tensors, indirect_expr=index, loop=loop)
        return load

    def check_bounds(self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool) -> None:
        asc_tensors = [s.asc_tensor for s in expr.free_symbols if isinstance(s, ASCIndexing)]
        ir.check_bounds(*asc_tensors, expr=expr, size=size, lower=lower, upper=upper)

    def index_to_str(self, index):
        return str(index)

    def _load_indirect_buffer(self, name):
        buf: ASCBuffer = self.get_asc_buffer(name)
        exist_tensor = self.graph.get_input_tensor(name)
        if exist_tensor is not None:
            return exist_tensor
        return buf.bind(self.graph.input(name, buf.dtype))

    def _load_buffer(self, name, loop: Loop):
        buf: ASCBuffer = self.get_asc_buffer(name)
        exist_tensor = self.graph.get_input_tensor(name)
        if exist_tensor is not None:
            return exist_tensor, loop
        return buf.bind(self.graph.input(name, buf.dtype)), loop

    def _store_buffer(self, name, value, loop: Loop):
        # 单图模式下，先记录本地值；如果当前 buffer 不属于 kernel 的最终输出，就完全跳过
        # Store/Output，避免无谓的 workspace 内存来回；属于最终输出时仍然 emit Store+Output。
        self._local_stores[name] = (value, loop.copy())
        if name not in self._outputs:
            return value
        store = ir.store(value, loop=loop)
        buf: ASCBuffer = self.get_asc_buffer(name)
        buf.bind(self.graph.output(name, buf.dtype, src=store))
        return store

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

    def _get_view_road(self, src: Loop, dst: DenseLoop):
        """求出从 src loop 形变到 dst loop 的 op 序列，按 list 顺序依次 apply。

        ascir 约束：**Load 以外的所有节点（Transpose / Broadcast / 算术 op…），
        输出的 size/stride 必须是 contiguous 关系**（stride[i] = product(size[i+1:]),
        size=1 维 stride=0）。所以 Transpose / Broadcast 的 .dst 一律用
        `DenseLoop(axis, size)` 重建，不复用 src 那个 non-contiguous 的 stride。

        src 是 `_index_to_loop` 给的："axis 按 dst.axis 顺序标，stride 反映上游
        buffer 的真实物理 layout"（permute/broadcast 视图叠加后给 inductor 看见的
        形态）；只有作为 Load.y 时才允许 non-contiguous。

        分两步：
          1. transpose：把 src.axis 按 stride 大→小重排出"contig 形态"。这是
             load 节点真实输出的视图（stride 单调递减）。Transpose op:
               src = contig 形态（= load.y）
               dst = DenseLoop(axis=dst.axis, size=src.size) 即 axis 还原到 dst
                     顺序、size 跟 src 一致、stride 重新算成 contiguous
             仅当 contig != src 才需要这步。
          2. broadcast：逐维把 size=1 升到 dst.size[dim]，每步 Broadcast op:
               src = 当前 contiguous loop
               dst = DenseLoop(axis, size_after) 升 size 后重算 contiguous stride
        """
        if src == dst:
            return []
        num_axis = len(src.axis)

        class MoveOp:
            def __init__(self, *, kind, src, dst):
                self.kind = kind
                self.src = src
                self.dst = dst

        # ---- step 1: 推 contig 形态 + 生成 Transpose op ----
        hint_to_axis = []
        for hint, axis, size, order in zip(src.hint_stride, src.axis, src.size, range(num_axis)):
            if hint != 0:
                hint_to_axis.append((hint, Axis(axis, size, order)))
        ordered_axis = [axis for _, axis in sorted(hint_to_axis, reverse=True)]
        non1_order = [axis.order for axis in ordered_axis]
        iter_non1_order = iter(non1_order)
        expect_dims = [i if i not in non1_order else next(iter_non1_order) for i in range(num_axis)]

        contig = src.copy()
        current_dims = list(range(num_axis))
        for i in reversed(range(num_axis)):
            if current_dims[i] != expect_dims[i]:
                j = current_dims.index(expect_dims[i])
                contig.transpose_(i, j)
                current_dims[i], current_dims[j] = current_dims[j], current_dims[i]

        road = []
        if contig != src:
            # Transpose.dst：axis 跟 dst 一致，size 跟 src 一致（broadcast 还没升），
            # stride 重新算成 contiguous（不复用 src 那个 non-contig stride）。
            transpose_dst = DenseLoop(axis=list(src.axis), size=list(src.size))
            road.append(MoveOp(kind="transpose", src=contig.copy(), dst=transpose_dst.copy()))
            cur = transpose_dst
        else:
            cur = src.copy()

        # ---- step 2: 逐维 broadcast 把 size=1 升到 dst.size[dim] ----
        broadcast_dims = [
            i
            for i, (s, d) in enumerate(zip(cur.size, dst.size))
            if str(s) == '1' and str(s) != str(d)
        ]
        for dim in broadcast_dims:
            new_size = list(cur.size)
            new_size[dim] = dst.size[dim]
            nxt = DenseLoop(axis=list(cur.axis), size=new_size)
            road.append(MoveOp(kind="broadcast", src=cur.copy(), dst=nxt.copy()))
            cur = nxt

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
    lines.extend(node.debug_str().split("\n"))
    lines = [v for v in lines if v]
    return lines


def _get_transposed_indexing(load_index, axis_vars):
    transposed_index = []
    for buffer, index in load_index.items():
        hints = V.graph.sizevars.stride_hints(index, axis_vars)
        non_zero_hints = [hint for hint in hints if str(hint) != '0']
        if sorted(non_zero_hints, reverse=True) != non_zero_hints:
            score = 1 if non_zero_hints[-1] == 1 else 100
            transposed_index.append((buffer, index, score))
    return transposed_index


def _fused_layout_tuple_py(parts: List[str]) -> str:
    """Python tuple literal for generated wrapper; keep (x,) for rank-1 / scalar metadata."""
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ", ".join(parts) + ")"


def _emit_fused_layout_checks(
    wrapper: PythonWrapperCodegen,
    kernel: NPUKernel,
    call_args,
    precompile_args,
    scheduling: "NPUScheduling",
) -> None:
    """
    Emit layout checks for kernel inputs (size/stride/dtype/device, skips storage_offset).
    Deduplicates by buffer name, each tensor checked only once.
    """
    input_outer_set = set(kernel.fused_graph.inputs_outer)
    input_call_args = {
        pa.buffer for pa in precompile_args
        if isinstance(pa, TensorArg) and pa.name in input_outer_set
    }
    seen_buffers: Set[str] = set()

    if not scheduling._fused_layout_import_emitted:
        wrapper.writeline(fused_layout_check.IMPORT_LINE)
        scheduling._fused_layout_import_emitted = True

    for buffer, buffer_type in zip(call_args, precompile_args):
        if not isinstance(buffer_type, TensorArg) or buffer not in input_call_args or buffer in seen_buffers:
            continue
        seen_buffers.add(buffer)

        buf = V.graph.get_buffer(buffer)
        layout = buf.layout
        stride = getattr(layout, "stride", None)
        if stride is None:
            logger.debug("skip layout check for buffer: no stride attribute")
            continue

        sz_py = [pycode(V.graph.sizevars.simplify(s)) for s in layout.size]
        st_py = [pycode(V.graph.sizevars.simplify(s)) for s in stride]

        from torch._inductor import config

        file_path = None
        if config.trace.enabled and hasattr(V.debug, "filename"):
            file_path = V.debug.filename('')
        wrapper.writeline(
            f"maybe_check_fused_input_layout("
            f"kernel_name={kernel.kernel_name!r}, buffer_name={buffer!r}, tensor={buffer}, "
            f"expected_sizes={_fused_layout_tuple_py(sz_py)}, "
            f"expected_strides={_fused_layout_tuple_py(st_py)}, "
            f"expected_dtype={repr(layout.dtype)}, expected_device_type={layout.device.type!r}, "
            f"path={file_path!r})"
        )


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._fuse_judge = TritonScheduling(scheduler)
        self._kernel_cache: Dict[str, NPUKernel] = dict()
        self._fused_layout_import_emitted: bool = False

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        return self._fuse_judge.can_fuse_horizontal(node1, node2)

    def group_fn(self, sizes):
        return self._fuse_judge.group_fn(sizes)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            BackendFeature.INPLACE_BUFFERS,
        ])

    def codegen_template(self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]):
        raise NotImplementedError()

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:
        self.codegen_nodes(node.get_nodes())

    def codegen_nodes(self, nodes: List[BaseSchedulerNode]):
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        wrapper: PythonWrapperCodegen = V.graph.wrapper_code
        comments = _node_comment(nodes)
        for comment in comments:
            wrapper.writeline(comment)

        logger.debug("Generating kernel for fused:\n%s", "\n".join(comments))
        kernel = NPUKernel(nodes, comments=comments).tracing_asc()

        arg_defs, call_args, precompile_args, arg_types = kernel.args.python_argdefs()

        kernel.kernel_name = kernel.fused_graph.name
        cache_hint = kernel.kernel_name
        cache_kernel = self._kernel_cache.get(cache_hint, None)
        if cache_kernel is not None:
            logger.debug("Reuse cached kernel %s for %s", cache_kernel.kernel_name, kernel.kernel_name)
            # 缓存命中仍用当前图的 buffer/layout 生成校验，调用的是缓存的 kernel 符号名。
            _emit_fused_layout_checks(wrapper, kernel, call_args, precompile_args, self)
            wrapper.writeline(wrapper.wrap_kernel_call(cache_kernel.kernel_name, [str(v) for v in call_args]))
            return

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

        # 在 wrap_kernel_call / ctypes 进入 C++ launch 之前做 Python 侧 layout 契约校验。
        _emit_fused_layout_checks(wrapper, kernel, call_args, precompile_args, self)
        wrapper.writeline(wrapper.wrap_kernel_call(kernel.kernel_name, [str(v) for v in call_args]))
        self._kernel_cache[cache_hint] = kernel

        from torch._inductor import config

        if config.trace.enabled:
            kernel.benchmark(nodes, V.debug.filename(f"{kernel.kernel_name}/benchmark.py"))
            kernel.view_dot(nodes, V.debug.filename(f"{kernel.kernel_name}/graph.svg"))
            kernel.record_summary(nodes, V.debug.filename(f"{kernel.kernel_name}/fuse_summary.csv"))

    def codegen_sync(self):
        raise NotImplementedError()

    def flush(self):
        pass

    def benchmark_fused_nodes(self, nodes):
        raise NotImplementedError()


class NpuWrapperCodeGen(PythonWrapperCodegen):
    @staticmethod
    def create(*args, **kwargs):
        wrapper_codegen = PythonWrapperCodegen.create(*args, **kwargs)
        wrapper_codegen.imports.splice(f"from {__package__}.compiler import async_compile as async_compile_ascendc")
        return wrapper_codegen
