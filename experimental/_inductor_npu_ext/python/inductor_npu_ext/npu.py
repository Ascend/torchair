# pylint: disable=W1203,E1125,W1514,R1729,W0246,W0201
import dataclasses
import itertools
import contextlib
import hashlib
import os
from typing import List, Dict, Union, Set, Optional
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
from .config import disable_cat_fuse, disable_canfuse, fuse_reduction_axis_threshold

if disable_cat_fuse:
    class NPUConcatBuffer:
        pass
else:
    from .lowering.cat_lowering import NPUConcatBuffer


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
        映射到这组 canonical 轴。
        如果 fused 组里有 NPUConcatBuffer，强制采用它的输出 sizes —— concat 轴必须是
        ΣM_i 的那一份，否则上游 partial 节点的 size 会比 canonical 还大。"""
        chosen_sizes: List[sympy.Expr] = []

        for node in self._nodes:
            buf = getattr(node, 'node', None)
            if isinstance(buf, NPUConcatBuffer):
                chosen_sizes = [self.rename_indexing(V.graph.sizevars.simplify(s))
                                for s in buf.get_size()]
                break

        if not chosen_sizes:
            for node in self._nodes:
                body: LoopBody = getattr(node, '_body', None)
                if body is None:
                    continue
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

    @staticmethod
    def _same_size(lhs, rhs) -> bool:
        return sympy.simplify(lhs - rhs) == 0

    def _local_zero_stride_axis_indices(self, canonical_axes) -> set:
        zero_stride_axis_indices = set()
        for _, loop in self._local_stores.values():
            if len(loop.axis) != len(canonical_axes):
                continue
            for i, (loop_axis, stride, size) in enumerate(zip(loop.axis, loop.stride, loop.size)):
                if str(loop_axis) != str(canonical_axes[i]):
                    continue
                if str(size) != "1" and sympy.simplify(stride) == 0:
                    zero_stride_axis_indices.add(i)
        return zero_stride_axis_indices

    def _find_axis_mapping_groups(self, node_sizes, canonical_ranges, canonical_axes):
        """Find ordered non-empty canonical axis groups for each node axis.

        Fused kernels can contain nodes with fewer axes than the canonical loop,
        for example a post-reduction consumer. In that case the missing canonical
        axes are broadcast axes for this node. Prefer skipping axes that prior
        local stores wrote with zero stride, since those are usually reduction
        axes removed from the consumer's logical shape.
        """
        num_canonical_axes = len(canonical_ranges)
        candidates: List[List[List[int]]] = []

        def search_axis_group(node_size, start_idx, selected_indices, product):
            for axis_idx in range(start_idx, num_canonical_axes):
                next_product = product * canonical_ranges[axis_idx]
                next_indices = selected_indices + [axis_idx]
                if self._same_size(next_product, node_size):
                    yield next_indices
                yield from search_axis_group(node_size, axis_idx + 1, next_indices, next_product)

        def search(node_idx: int, canonical_idx: int, groups: List[List[int]]):
            if node_idx == len(node_sizes):
                candidates.append(groups.copy())
                return

            for group_indices in search_axis_group(node_sizes[node_idx], canonical_idx, [], sympy.S.One):
                groups.append(group_indices)
                search(node_idx + 1, group_indices[-1] + 1, groups)
                groups.pop()

        search(0, 0, [])
        if not candidates:
            return None

        zero_stride_axis_indices = self._local_zero_stride_axis_indices(canonical_axes)

        def score(groups: List[List[int]]):
            used_indices = {
                axis_idx
                for group_indices in groups
                for axis_idx in group_indices
            }
            skipped_indices = set(range(num_canonical_axes)) - used_indices
            skipped_zero_stride = len(skipped_indices & zero_stride_axis_indices)
            skipped_nonzero_stride = len(skipped_indices - zero_stride_axis_indices)
            # Prefer preserving trailing alignment when size equality is ambiguous.
            later_axis_alignment = sum(group_indices[0] for group_indices in groups)
            flatten_penalty = sum(len(group_indices) - 1 for group_indices in groups)
            return (
                skipped_zero_stride,
                later_axis_alignment,
                -skipped_nonzero_stride,
                -flatten_penalty,
            )

        return max(candidates, key=score)

    def _node_axis_indexings(self, node, canonical_axes, canonical_ranges):
        """`_align_node_to_canonical` 的输出之一，单独保留 thin wrapper 便于复用。"""
        return self._align_node_to_canonical(node, canonical_axes, canonical_ranges)[0]

    def _align_node_to_canonical(self, node, canonical_axes, canonical_ranges,
                                 partial_axis_sizes: Optional[List[sympy.Expr]] = None):
        """把 node 的轴对齐到 canonical 轴。三种模式：
        - **1:1**（节点轴数 == canonical 轴数）：每个 node 轴直接对应一个 canonical 轴，
          *不要求 size 相等*。覆盖常规 pointwise + concat 上游 pointwise 的简单场景。
        - **flatten**（节点轴数 < canonical 轴数）：单个 node 轴展开成多个连续 canonical
          轴的 contiguous 多项式（pointwise collapse）。剩余未映射的 canonical 轴等价
          于 broadcast，size=1。
        - **partial flatten**（concat prologue 专用）：node 把 [N, M_i] flatten 成
          单 axis size = N*M_i。这时 canonical 是 [N, ΣM_i]，product 对不齐 ΣM_i。
          调用方传 `partial_axis_sizes = [N, M_i]`，按 partial size 做 flatten，
          表达式形如 `a0*M_i + a1`。

        返回 (axis_indexings, node_canonical_sizes)。
        """
        body: LoopBody = getattr(node, '_body')
        transpose_order = self._get_minimal_transpose_order(node)
        node_sizes = [self.rename_indexing(body.var_ranges[v]) for v in transpose_order]

        var_to_expr: Dict[sympy.Symbol, sympy.Expr] = {}
        node_canonical_sizes: List[sympy.Expr] = [sympy.S.One] * len(canonical_axes)

        if len(node_sizes) == len(canonical_axes):
            # 1:1 模式
            for i, (node_var, axis, sz) in enumerate(
                    zip(transpose_order, canonical_axes, node_sizes)):
                var_to_expr[node_var] = axis
                node_canonical_sizes[i] = sz
        elif (partial_axis_sizes is not None
              and len(node_sizes) < len(canonical_axes)
              and len(partial_axis_sizes) == len(canonical_axes)):
            # partial flatten：用 partial_axis_sizes 来对齐
            canonical_idx = 0
            for node_var, node_size in zip(transpose_order, node_sizes):
                group_axes: List[sympy.Symbol] = []
                group_sizes: List[sympy.Expr] = []
                product = sympy.S.One
                start_idx = canonical_idx
                while canonical_idx < len(canonical_axes):
                    group_axes.append(canonical_axes[canonical_idx])
                    group_sizes.append(partial_axis_sizes[canonical_idx])
                    product = product * partial_axis_sizes[canonical_idx]
                    node_canonical_sizes[canonical_idx] = partial_axis_sizes[canonical_idx]
                    canonical_idx += 1
                    if sympy.simplify(product - node_size) == 0:
                        break
                if sympy.simplify(product - node_size) != 0:
                    raise RuntimeError(
                        f"Cannot map node axis {node_var}(size={node_size}) using "
                        f"partial sizes {partial_axis_sizes}")
                var_to_expr[node_var] = self._flatten_expr(group_axes, group_sizes)
        else:
            mapping_groups = self._find_axis_mapping_groups(node_sizes, canonical_ranges, canonical_axes)
            if mapping_groups is None:
                raise RuntimeError(
                    f"Cannot map node axes with sizes {node_sizes} into canonical axes "
                    f"with sizes {canonical_ranges}")

            for node_var, group_indices in zip(transpose_order, mapping_groups):
                group_axes = [canonical_axes[idx] for idx in group_indices]
                group_sizes = [canonical_ranges[idx] for idx in group_indices]
                var_to_expr[node_var] = self._flatten_expr(group_axes, group_sizes)
                for k in group_indices:
                    node_canonical_sizes[k] = canonical_ranges[k]

        axis_indexings: List[List[sympy.Expr]] = []
        for var in body.var_ranges.keys():
            expr = var_to_expr.get(var)
            if expr is None:
                expr = sympy.Symbol(var.name)
            axis_indexings.append([expr])
        return axis_indexings, node_canonical_sizes

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

            # 找出本 kernel 里的 NPUConcatBuffer（最多一个），用来识别 prologue + 推 partial sizes
            concat_buf: Optional[NPUConcatBuffer] = None
            concat_input_shapes: Dict[str, List[sympy.Expr]] = {}
            for n in self._nodes:
                cb = getattr(n, 'node', None)
                if isinstance(cb, NPUConcatBuffer):
                    concat_buf = cb
                    for inp, input_layout in zip(cb.inputs, cb.input_layouts):
                        concat_input_shapes[inp.get_name()] = [
                            self.rename_indexing(V.graph.sizevars.simplify(s))
                            for s in input_layout.size
                        ]
                    break

            prior_loop = self._current_loop
            try:
                for i, node in enumerate(self._nodes):
                    logger.debug("Codegen [%s] %s", f"{i+1}/{len(self._nodes)}", node.debug_str())
                    buf = getattr(node, 'node', None)
                    if isinstance(buf, NPUConcatBuffer):
                        # NPUConcatBuffer 没 body，绕开 node.run；直接调 kernel.concat
                        self._current_loop = canonical_loop  # 输出取 canonical 全 size
                        prior_node = self.current_node
                        self.current_node = node  # 跳 set_current_node（依赖 _body.bounds）
                        try:
                            node.mark_run()  # 触发 wrapper.codegen_allocation 给输出 buffer
                            self.concat(buf)
                        finally:
                            self.current_node = prior_node
                    else:
                        # 检测 prologue：node 输出在 NPUConcatBuffer 的 inputs 列表里。
                        # 真实 shape 取那个 input buffer 的 size（concat 轴是 partial）。
                        partial_sizes = None
                        if concat_buf is not None:
                            for out in getattr(node, 'outputs', []):
                                shape = concat_input_shapes.get(out.node.name)
                                if shape is not None:
                                    partial_sizes = shape
                                    break
                        axis_indexings, node_sizes = self._align_node_to_canonical(
                            node, canonical_axes, canonical_ranges,
                            partial_axis_sizes=partial_sizes)
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
        dtype = self.get_asc_buffer(name).dtype

        # 本地融合：当前 fused kernel 内已经 store 过 name，直接复用值，避免在同一图中
        # 同时存在 Data 和 Output 形成 FusedGraph 自环。
        if name in self._local_stores:
            value, src_loop = self._local_stores[name]
            if dtype in {torch.bfloat16, torch.float16}:
                src_loop = src_loop.copy().contiguous_()
                value = ir.cast(value, dst=torch.float32, loop=src_loop)
            return self._reshape_local_value(value, src_loop, sizes)

        data, loop = self._load_buffer(name, self._index_to_loop(index, sizes=sizes))
        offset = loop.zero_offset_()
        road = self._get_view_road(loop, DenseLoop(axis=loop.axis, size=sizes))

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

    def concat(self, concat_buffer: 'NPUConcatBuffer'):
        """把 NPUConcatBuffer 翻译成一条 ascir.Concat。
        - 如果上游 pointwise 已被融到本 kernel（_local_stores 命中），直接复用它产出的
          partial 张量——无需再起一个 Data/Load；
        - 否则按 input buffer 起 Data + Load，loop 取 partial（concat 轴 size = input 自己的）；
        - Concat 输出取 canonical 全 size，由 _store_buffer 落盘到输出 buffer。
        """
        axis = concat_buffer.axis
        out_name = concat_buffer.get_name()

        loaded = []
        for buf, input_layout in zip(concat_buffer.inputs, concat_buffer.input_layouts):
            in_name = buf.get_name()
            if in_name in self._local_stores:
                # 上游融合命中：取已经在 partial loop 上算好的张量
                value, _src_loop = self._local_stores[in_name]
                loaded.append(value)
                continue
            asc_in = ASCBuffer(in_name, input_layout)
            partial_loop = self._make_partial_loop(asc_in, axis)
            data, _ = self._load_buffer(in_name, partial_loop)
            offset = partial_loop.zero_offset_()
            load = ir.load(data, offset=offset, loop=partial_loop)
            loaded.append(load)

        out = ir.concat(loaded, axis=axis)
        self._store_buffer(out_name, out, self.contiguous_loop)

    def _make_partial_loop(self, asc_in: 'ASCBuffer', concat_axis: int) -> Loop:
        """input 的 loop：axis names 跟 canonical 对齐，concat 轴 size 用本 input 自己的，
        其它 axis 用 canonical 的。stride 按 input 自身 layout 推。"""
        canonical = self.contiguous_loop
        size = []
        for i, c_size in enumerate(canonical.size):
            if i == concat_axis:
                size.append(asc_in.size[i])
            else:
                size.append(c_size)
        loop = Loop()
        loop.axis = list(canonical.axis)
        loop.size = size
        loop.stride = list(asc_in.stride)
        loop.offset = asc_in.offset
        return loop

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
        store = ir.store(name, value, loop=loop)
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

    @staticmethod
    def _find_concat_buffer(node: BaseSchedulerNode) -> Optional['NPUConcatBuffer']:
        """从普通 SchedulerNode 或 FusedSchedulerNode 里抠出 NPUConcatBuffer，没有就返回 None。"""
        for sub in node.get_nodes():
            buf = getattr(sub, 'node', None)
            if isinstance(buf, NPUConcatBuffer):
                return buf
        return None

    @staticmethod
    def _node_output_names(node: BaseSchedulerNode):
        names = []
        for sub in node.get_nodes():
            for out in getattr(sub, 'outputs', []):
                names.append(out.node.name)
        return names

    @staticmethod
    def _concat_output_names(node: BaseSchedulerNode):
        names = []
        for sub in node.get_nodes():
            if not isinstance(getattr(sub, 'node', None), NPUConcatBuffer):
                continue
            for out in getattr(sub, 'outputs', []):
                names.append(out.node.name)
        return names

    @classmethod
    def _is_concat_epilogue(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        concat_outputs = set(cls._concat_output_names(producer))
        if not concat_outputs:
            return False
        consumer_rw = getattr(consumer, 'read_writes', None)
        if consumer_rw is None:
            return False
        consumer_reads = {dep.name for dep in consumer_rw.reads}
        return bool(concat_outputs & consumer_reads)

    @classmethod
    def _is_concat_prologue(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        """producer 直接喂给 consumer 内某个 NPUConcatBuffer 的某路输入，就允许融合。
        注意 consumer 可能是 FusedSchedulerNode（NPUConcatBuffer 已经先跟其它 prologue
        融过一轮），需要递归找内部的 NPUConcatBuffer。"""
        cb = cls._find_concat_buffer(consumer)
        if cb is None:
            return False
        cb_input_names = {inp.get_name() for inp in cb.inputs}
        for name in cls._node_output_names(producer):
            if name in cb_input_names:
                return True
        return False

    @classmethod
    def _share_concat(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        """两个节点是不是同一个 NPUConcatBuffer 的兄弟 input —— 用于水平融合。"""
        for n1_name in cls._node_output_names(node1):
            for sub in [node1, node2]:
                for s in sub.get_nodes():
                    for o in getattr(s, 'outputs', []):
                        for u in o.users:
                            cb = cls._find_concat_buffer(u.node) if hasattr(u.node, 'get_nodes') else (
                                u.node.node if isinstance(getattr(u.node, 'node', None), NPUConcatBuffer) else None
                            )
                            if cb is None:
                                continue
                            cb_inputs = {inp.get_name() for inp in cb.inputs}
                            n1_outs = set(cls._node_output_names(node1))
                            n2_outs = set(cls._node_output_names(node2))
                            if n1_outs & cb_inputs and n2_outs & cb_inputs:
                                return True
        return False

    @staticmethod
    def _contiguous_strides_for_sizes(sizes):
        strides = []
        running = sympy.S.One
        for size in reversed(sizes):
            strides.append(sympy.S.Zero if str(size) == "1" else running)
            running = running * size
        return list(reversed(strides))

    @classmethod
    def _memory_dep_is_contiguous_write(cls, dep) -> bool:
        if dep.is_contiguous():
            return True
        try:
            if V.graph.sizevars.simplify(dep.get_offset()) != 0:
                return False
            actual_strides = V.graph.sizevars.stride_vars(dep.index, dep.var_names)
            expected_strides = cls._contiguous_strides_for_sizes(dep.size)
            return all(
                V.graph.sizevars.simplify(actual - expected) == 0
                for actual, expected in zip(actual_strides, expected_strides)
            )
        except Exception:
            return False

    @classmethod
    def _vertical_outputs_are_contiguous(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        producer_rw = getattr(producer, 'read_writes', None)
        consumer_rw = getattr(consumer, 'read_writes', None)
        if producer_rw is None or consumer_rw is None:
            return True
        producer_writes = {
            dep.name: dep
            for dep in producer_rw.writes
        }
        consumer_reads = {
            dep.name
            for dep in consumer_rw.reads
        }
        shared_names = producer_writes.keys() & consumer_reads
        return all(cls._memory_dep_is_contiguous_write(producer_writes[name]) for name in shared_names)

    @staticmethod
    def _reduction_axis_numel(node: BaseSchedulerNode):
        sizes = getattr(node, '_sizes', None)
        if not sizes or len(sizes) < 2:
            return None
        reduction_sizes = sizes[1]
        if not reduction_sizes:
            return None
        numel = sympy.S.One
        for size in reduction_sizes:
            numel = numel * V.graph.sizevars.simplify(size)
        return V.graph.sizevars.simplify(numel)

    @staticmethod
    def _is_reduction_axis_below_fuse_threshold(numel) -> bool:
        numel = V.graph.sizevars.simplify(numel)
        try:
            return int(numel) < fuse_reduction_axis_threshold
        except (TypeError, ValueError):
            pass

        try:
            return bool(V.graph.sizevars.evaluate_expr(
                sympy.Lt(numel, fuse_reduction_axis_threshold),
                fallback_value=False,
            ))
        except Exception:
            return False

    @classmethod
    def _reduction_axis_within_fuse_threshold(cls, node: BaseSchedulerNode) -> bool:
        if fuse_reduction_axis_threshold < 0:
            return True
        for subnode in node.get_nodes():
            if not subnode.is_reduction():
                continue
            numel = cls._reduction_axis_numel(subnode)
            if numel is None:
                return False
            if not cls._is_reduction_axis_below_fuse_threshold(numel):
                return False
        return True

    @classmethod
    def _reduction_axes_within_fuse_threshold(cls, *nodes: BaseSchedulerNode) -> bool:
        return all(cls._reduction_axis_within_fuse_threshold(node) for node in nodes)

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if disable_canfuse:
            return False
        if self._is_concat_epilogue(node1, node2):
            return False
        if not self._vertical_outputs_are_contiguous(node1, node2):
            return False
        if not self._reduction_axes_within_fuse_threshold(node1, node2):
            return False
        if self._is_concat_prologue(node1, node2):
            return True
        return self._fuse_judge.can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if disable_canfuse:
            return False
        if not self._reduction_axes_within_fuse_threshold(node1, node2):
            return False
        if self._share_concat(node1, node2):
            return True
        return self._fuse_judge.can_fuse_horizontal(node1, node2)

    def group_fn(self, sizes):
        return self._fuse_judge.group_fn(sizes)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            BackendFeature.INPLACE_BUFFERS,
        ])

    def codegen_template(
            self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode],
            prologue_nodes: List[SchedulerNode] = (),
    ):
        # 目前只接 NPUConcatBuffer 一种 template。epilogue 还不支持，prologue 直接
        # 跟 template 一起塞进 codegen_nodes —— tracing_asc 的主循环已能识别
        # NPUConcatBuffer 节点并走 kernel.concat（其它节点照常 node.run）。
        if not isinstance(getattr(template_node, 'node', None), NPUConcatBuffer):
            raise NotImplementedError(
                f"unknown template buffer: {type(template_node.node).__name__}")
        if epilogue_nodes:
            raise NotImplementedError("NPUConcatBuffer + epilogue fusion not yet supported")
        self.codegen_nodes(list(prologue_nodes) + [template_node])

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
