import dataclasses
import itertools
import contextlib
import hashlib
import os
from typing import List, Dict, Union, Set
from collections import OrderedDict

import sympy
import torch

from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.ir import LoopBody
from torch._inductor.scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode, FusedSchedulerNode
from torch._inductor.utils import get_kernel_metadata, get_fused_kernel_name
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import IndentedBuffer, Kernel

from inductor_npu_ext.common import logger
from inductor_npu_ext.common.asc_graph import ASCGraph, FusedASCGraph, ASCIndexing
from inductor_npu_ext.common.symbols import Axis
from inductor_npu_ext.common.debug import _left_align_lines, OP_SUMMARY
from inductor_npu_ext.common.symbols import AscExpr, Loop, DenseLoop
from inductor_npu_ext.common.asc_graph import _Tensor, _Scalar
from inductor_npu_ext import asc_ops as ir
from inductor_npu_ext.asc_overrides import NPUOverrides
from inductor_npu_ext.config import _debug_options


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
        self._subgraphs: List[ASCGraph] = []
        self._indirect_to_scalar: Dict[str, _Scalar] = dict()
        self._current_loop = None
        self._asc_buffer: Dict[str, ASCBuffer] = {}
        self._torch_arg_wrappers = dict()
        self._nodes = nodes
        self._outputs = _get_nodes_outputs(nodes)

    @property
    def graph(self):
        return self._subgraphs[-1]

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
            index_transposed = _get_transposed_indexing(body.indexing_exprs, axis_vars)  # dict[idx:expr]读写内存使用的index
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

    @contextlib.contextmanager
    def new_subgraph(self, asc_axis: List[sympy.Symbol], asc_axis_range: List[sympy.Expr], *,
                     hint_str=None):
        if not asc_axis:
            asc_axis = [sympy.Symbol("z0")]
            asc_axis_range = [1]
        loop = DenseLoop(axis=asc_axis, size=asc_axis_range)
        self._subgraphs.append(ASCGraph(name=f"graph{len(self._subgraphs)}", hint_str=hint_str))
        self.graph.set_current_loop(loop)
        for axis, axis_range in zip(asc_axis, asc_axis_range):
            self.graph.axis(axis.name, axis_range)
        prior = self._current_loop
        self._current_loop = loop
        try:
            yield
        finally:
            self._current_loop = prior

    def tracing_asc(self):
        with self:
            for i, node in enumerate(self._nodes):
                logger.debug("Codegen [%s] %s", f"{i+1}/{len(self._nodes)}", node.debug_str())
                body: LoopBody = getattr(node, '_body')

                var_to_asc_axis = {}
                axis_indexings = []
                for var in body.var_ranges.keys():
                    var_to_asc_axis[var] = sympy.Symbol(var.name)
                    axis_indexings.append([var_to_asc_axis[var]])

                asc_axis = []
                asc_axis_range = []
                for var in self._get_minimal_transpose_order(node):
                    asc_axis.append(var_to_asc_axis[var])
                    asc_axis_range.append(self.rename_indexing(body.var_ranges[var]))

                with self.set_current_node(node), self.new_subgraph(asc_axis, asc_axis_range,
                                                                    hint_str='\n'.join(_node_label(node))):
                    node.run(*axis_indexings)
                    logger.debug(f"{self.graph.name} reads {self.graph.inputs} and writes {self.graph.outputs}")

        if hasattr(self, 'removed_buffers') and hasattr(V.graph, 'removed_buffers'):
            V.graph.removed_buffers |= self.removed_buffers
        if hasattr(self, 'inplaced_to_remove') and hasattr(V.graph, 'inplaced_to_remove'):
            V.graph.inplaced_to_remove |= self.inplaced_to_remove

        for subgraph in self._subgraphs:
            for sym, sym_renamed in self.args.sizevars.items():
                subgraph.size(sym_renamed)

        self._fused_graph = FusedASCGraph(subgraphs=self._subgraphs, outputs=self._outputs)
        # 对于输出复用输入的场景，可能出现多个asc graph上的buffer（Data/Output）对应同一个python kernel入参的情况，
        # outer是python kernel层的入参名，而inputs/outputs，则是asc graph上的buffer名，也对应rt层kernel的args
        self._fused_graph.inputs_outer = [self.args.input(read) for read in self._fused_graph.inputs]
        self._fused_graph.outputs_outer = [self.args.output(write) for write in self._fused_graph.outputs]
        # 这里的args，对应python kernel签名的入参名字，也是wrapper签名中的入参名字。
        # 而第二个返回，是在output code call函数中，调用python kernel时传入的参数，也就是实际buffer的名字。
        arg_defs, call_args, precompile_args, arg_types = self.args.python_argdefs()
        self._fused_graph.args = precompile_args

        from inductor_npu_ext import codegen as npu_codegen
        self._fused_graph.cpp_wrapper = npu_codegen.codegen_cpp_wrapper(self._fused_graph)
        self._fused_graph.asc_graph = self._fused_graph.codegen("cache_hint").getvalue()
        md5 = hashlib.md5(f"{self._fused_graph.asc_graph}_{self._fused_graph.cpp_wrapper}".encode()).hexdigest()
        self._fused_graph.name = f"auto{get_fused_kernel_name(self._nodes, 'original_aten')}_{md5}"

        unsupported_ops = set()
        for subgraph in self._subgraphs:
            unsupported_ops |= subgraph.unsupported_ops

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
        from inductor_npu_ext import codegen as npu_codegen
        artifacts = npu_codegen.codegen_kernel_def(self.fused_graph)
        artifacts['cpp_wrapper'] = npu_codegen.codegen_cpp_wrapper(self.fused_graph)
        if not all(v.strip() for v in artifacts.values()):
            raise RuntimeError(f"Failed to generate artifacts for kernel {self.kernel_name}: {artifacts}")

        self._artifacts = NPUKernel.Artifacts(**artifacts)

        kernel_def = IndentedBuffer()
        kernel_obj = f"{self._artifacts.name}_artifacts"
        kernel_def.writeline(f"{kernel_obj} = {{}}")
        kernel_def.splice(f"{kernel_obj}['name'] = r'''{self._artifacts.name}'''")
        kernel_def.splice(f"{kernel_obj}['tiling_def'] = r'''{self._artifacts.tiling_def}'''")
        kernel_def.splice(f"{kernel_obj}['host_impl'] = r'''{self._artifacts.host_impl}'''")
        kernel_def.splice(f"{kernel_obj}['device_impl'] = r'''{self._artifacts.device_impl}'''")
        kernel_def.splice(f"{kernel_obj}['cpp_wrapper'] = r'''{self._artifacts.cpp_wrapper}'''")
        kernel_def.writeline(
            f"{self.kernel_name} = async_compile_ascendc(globals().get('async_compile', None), {kernel_obj})")

        return kernel_def.getvalue()

    def record_summary(self, nodes, model_path=None):
        for i, graph in enumerate(self._subgraphs):
            loop_body = _node_label(nodes[i]) if i < len(nodes) else ""
            OP_SUMMARY.add_graph_summary(graph, loop=loop_body, model_path=model_path)

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
                pydot.Node(f"{self.kernel_name}_body", shape="plaintext", label='\n'.join(lines),
                           fontname="Courier"))
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
            from torch._inductor.codegen.common import TensorArg
            if not isinstance(buffer_type, TensorArg):
                continue
            used_buffers.append(buffer)
            layout = V.graph.get_buffer(buffer).layout
            for expr in itertools.chain(layout.stride or [], layout.size or [], [layout.offset]):
                seen_symbols.extend(V.graph.sizevars.simplify(
                    expr).free_symbols if isinstance(expr, sympy.Expr) else [])

        with open(file_path, "w") as f:
            benchmark_code = IndentedBuffer()
            benchmark_code.writeline(f"import sys")
            benchmark_code.writeline(f"import torch")
            benchmark_code.writeline(f"import torch_npu")
            benchmark_code.writeline(
                "from inductor_npu_ext.compiler import async_compile as async_compile_ascendc")
            kernel_obj = f"{self._artifacts.name}_artifacts"
            benchmark_code.writeline(f"{kernel_obj} = {{}}")
            benchmark_code.splice(f"{kernel_obj}['name'] = r'''{self._artifacts.name}'''")
            benchmark_code.splice(f"{kernel_obj}['cpp_wrapper'] = r'''{self._artifacts.cpp_wrapper}'''")

            benchmark_code.writelines(["\n"] * 2)
            benchmark_code.writeline("if __name__ == '__main__':")
            with benchmark_code.indent():
                benchmark_code.writeline(
                    f"assert len(sys.argv) == 1 or sys.argv[-1] == 'e2e', 'Usage: python {file_path} [e2e]'")
                benchmark_code.writeline(f"if sys.argv[-1] == 'e2e':")
                with benchmark_code.indent():
                    with open(os.path.join(os.path.dirname(file_path), "asc_graph.py"), "r") as asc_graph:
                        benchmark_code.splice(asc_graph.read())
                    benchmark_code.splice(f"{kernel_obj}['tiling_def'] = tiling_def")
                    benchmark_code.splice(f"{kernel_obj}['host_impl'] = host_impl")
                    benchmark_code.splice(f"{kernel_obj}['device_impl'] = device_impl")
                benchmark_code.writeline(f"else:")
                with benchmark_code.indent():
                    benchmark_code.splice(f"{kernel_obj}['tiling_def'] = r'''{self._artifacts.tiling_def}'''")
                    benchmark_code.splice(f"{kernel_obj}['host_impl'] = r'''{self._artifacts.host_impl}'''")
                    benchmark_code.splice(f"{kernel_obj}['device_impl'] = r'''{self._artifacts.device_impl}'''")

                benchmark_code.writeline(f"{self.kernel_name} = async_compile_ascendc(None, {kernel_obj})")
                benchmark_code.writeline(f"from torch._dynamo.testing import rand_strided")
                for s, ks in self.args.sizevars.items():
                    benchmark_code.writeline(f"{ks} = {s} = {self.size_hint(s)}")
                for k in seen_symbols:
                    if k not in self.args.sizevars.keys():
                        benchmark_code.writeline(f"{k} = {self.size_hint(k)} # buffer size hint")
                for buffer in used_buffers:
                    layout = V.graph.get_buffer(buffer).layout
                    benchmark_code.writeline(
                        f"{buffer} = rand_strided({tuple(layout.size)}, {tuple(layout.stride)}, "
                        f"device='{layout.device if layout.device.type != 'npu' else 'npu'}', dtype={layout.dtype})")

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
                        benchmark_code.writeline(f"prof.step()")
            f.write(benchmark_code.getvalue())

    def load(self, name: str, index: sympy.Expr):
        if any([isinstance(s, ASCIndexing) for s in index.free_symbols]):
            return self.indirect_load(name, index)

        index = self.rename_indexing(index)
        sizes = self.contiguous_loop.size

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

    def store(self, name, index, value, mode=None):
        index = self.rename_indexing(index)
        dtype = self.get_asc_buffer(name).dtype
        loop = self._index_to_loop(index)
        if dtype in {torch.bfloat16, torch.float16}:
            value = ir.cast(value, dst=torch.float32, loop=loop)
            value = ir.cast(value, dst=dtype, loop=loop)
        store = ir.store(value, loop=loop)
        self._store_buffer(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

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
        store = ir.store(reduction, loop=loop)
        self._store_buffer(name, store)
        self.cse.store_cache.pop(name)  # Inductor cse always cache value, but we don't want to cache it
        return store

    def rename_indexing(self, index) -> sympy.Expr:
        return super().rename_indexing(index)

    def indirect_load(self, name: str, index: sympy.Expr) -> _Tensor:
        data, loop = self._load_buffer(name, self.contiguous_loop)
        asc_tensors = [s.asc_tensor for s in index.free_symbols if isinstance(s, ASCIndexing)]
        load = ir.indirect_load(data, *asc_tensors, indirect_expr=index, loop=loop)
        return load

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
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

    def _store_buffer(self, name, src):
        buf: ASCBuffer = self.get_asc_buffer(name)
        return buf.bind(self.graph.output(name, buf.dtype, src=src))

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
        expect_dims = [i if i not in non1_order else next(iter_non1_order) for i in range(num_axis)]

        class MoveOp:
            def __init__(self, *, kind, src, dst):
                self.kind = kind
                self.src = src
                self.dst = dst

        road = []

        current_dims = list(range(num_axis))
        src_loop = src.copy()
        for i in reversed(range(num_axis)):
            if current_dims[i] != expect_dims[i]:
                j = current_dims.index(expect_dims[i])
                src_loop.transpose_(i, j)
                current_dims[i], current_dims[j] = current_dims[j], current_dims[i]

        if src_loop != src:
            road.insert(0, MoveOp(kind="transpose", src=src_loop, dst=dst))

        road_dst = road[0].src if road else dst
        broadcast_dims = [i for i, (src_size, dst_size) in enumerate(zip(src_loop.size, road_dst.size))
                          if str(src_size) == '1' and str(src_size) != str(dst_size)]
        for dim in broadcast_dims:
            road_dst = road[0].src if road else dst
            road.insert(0, MoveOp(kind="broadcast", src=road_dst.copy().debroadcast_(dim), dst=road_dst))

        if len(road) > 0:
            road[0].src = src_loop.copy()

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


class NPUScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._fuse_judge = TritonScheduling(scheduler)
        self._kernel_cache: Dict[str, NPUKernel] = dict()

    def can_fuse_vertical(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if "canfuse" in _debug_options:
            return self._fuse_judge.can_fuse_vertical(node1, node2)

        return False  # disable until reliable evaluation algorithm is implemented.

    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        if "canfuse" in _debug_options:
            return self._fuse_judge.can_fuse_horizontal(node1, node2)

        return False  # disable until reliable evaluation algorithm is implemented.

    def group_fn(self, sizes):
        return self._fuse_judge.group_fn(sizes)

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT, BackendFeature.INPLACE_BUFFERS])

    def codegen_template(
            self, template_node: SchedulerNode, epilogue_nodes: List[SchedulerNode]
    ):
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
            wrapper.writeline(wrapper.wrap_kernel_call(cache_kernel.kernel_name, [str(v) for v in call_args]))
            return

        wrapper.header.splice("\n\n")
        wrapper.header.splice(kernel.codegen())

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
        wrapper_codegen.imports.splice(
            "from inductor_npu_ext.compiler import async_compile as async_compile_ascendc")
        return wrapper_codegen
