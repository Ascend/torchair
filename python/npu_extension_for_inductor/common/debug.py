import atexit
import functools
import operator
import os
from collections import defaultdict
import contextlib
from typing import Iterable, Dict, List, Set, Union

from npu_extension_for_inductor.common.asc_graph import ASCGraph, FusedASCGraph
from npu_extension_for_inductor.common.utils import StrRep
from npu_extension_for_inductor.ir import _Op, _Tensor


class _OpVisitor:
    def __init__(self, op):
        self.op = op
        self.inputs = [v for v in self.op.attrs.values() if isinstance(v, _Tensor)]

    @property
    def num_inputs(self):
        return len(self.inputs)

    def output_size(self, name='y'):
        return self.op.attrs[f'{self.op.name}.{name}.size']

    def output_stride(self, name='y'):
        return self.op.attrs[f'{self.op.name}.{name}.strides']

    @property
    def op_type(self):
        return self.op.op_type

    @property
    def compute_type(self):
        return self.op.get_private_attr("compute_type")


class _GraphVisitor:
    def __init__(self, graph: ASCGraph):
        self.graph = graph
        self.buffers = []
        self.workspaces = []
        self.load_ops = []
        self.store_ops = []
        self.reduction_ops = []
        self.pointwise_ops = []
        self.ops = [_OpVisitor(op) for op in self.graph.ops]
        self.num_load_no_fuse = 0
        self.num_store_no_fuse = 0
        self.typed_ops_count = defaultdict(lambda: 0)
        for op in self.ops:
            self.typed_ops_count[op.op_type] += 1
            if op.op_type in ["Data", "Output"]:
                self.buffers.append(op)
            elif op.op_type == "Workspace":
                self.workspaces.append(op)
            elif op.op_type == "Load":
                self.load_ops.append(op)
            elif op.op_type == "Store":
                self.store_ops.append(op)
            else:
                self.num_load_no_fuse += op.num_inputs
                self.num_store_no_fuse += 1
                if op.compute_type == 'reduce':
                    self.reduction_ops.append(op)
                else:
                    self.pointwise_ops.append(op)
        self.num_load = len(self.load_ops)
        self.num_store = len(self.store_ops)
        self.num_load_optimized = self.num_load_no_fuse - self.num_load
        self.num_store_optimized = self.num_store_no_fuse - self.num_store
        self.graph_type = "Fused" if len(self.workspaces) else "Reduction" if len(self.reduction_ops) else "PointWise"


class OpSummary:
    def __init__(self, graph, *, loop="", model_path=None):
        self.graph = ASCGraph(name=graph) if isinstance(graph, str) else graph
        self.loop = loop
        self.name = self.graph.name
        self.fallbacks: Dict[str:int] = defaultdict(lambda: 0)
        self.fallback_reasons: Dict[str:int] = defaultdict(lambda: "unknown")
        self.supported: Dict[str:int] = defaultdict(lambda: 0)
        self.unsupported: Dict[str:int] = defaultdict(lambda: 0)
        self.subs: Dict[str:OpSummary] = {}
        self.models: Set[str] = set()
        self.model_path = model_path
        self.num_supported: int = 0
        self.num_unsupported: int = 0
        self.calls: Dict[str:int] = defaultdict(lambda: 0)
        self.calls_detail: Dict[str:int] = defaultdict(lambda: 0)
        for op in self.graph.ops:
            self.record(op.op_type, op.supported)

    def record(self, op, is_supported):
        if is_supported:
            self.num_supported += 1
            self.supported[op] += 1
        else:
            self.num_unsupported += 1
            self.unsupported[op] += 1

    def fallback(self, op: str, reason: str):
        self.fallbacks[op] += 1
        self.fallback_reasons[op] = reason

    def add_graph_summary(self, graph, *, loop, model_path=None):
        if graph.name not in self.subs:
            self.subs[graph.name] = OpSummary(graph, loop=loop, model_path=model_path)
            if model_path:
                self.models.add(model_path)

        return self.subs[graph.name]

    def get_graph_summary(self, name):
        if name not in self.subs:
            return None
        return self.subs[name]

    def record_call_args(self, *args, sym_vals):
        axis_hint_sizes = {}
        for axis, axis_size in self.graph.axis_vars.items():
            hint_axis_size = eval(f"{axis_size}", dict((str(s), v) for s, v in zip(self.graph.size_vars, sym_vals)))
            axis_hint_sizes[axis] = hint_axis_size
        loop_size = functools.reduce(operator.mul, axis_hint_sizes.values(), 1)
        axis_hint_sizes['loop_size'] = loop_size
        args_key = ','.join(
            [f"{k}={v}" for k, v in list(zip(self.graph.size_vars, sym_vals)) + list(axis_hint_sizes.items())])
        self.calls[loop_size] += 1
        self.calls_detail[args_key] += 1

    def save(self):
        from torch._inductor import config
        if not config.trace.enabled:
            return

        self.save_csv(f"{self.name}.csv")
        print(str(self))

    @contextlib.contextmanager
    def open_model_csv_writers(self):
        import csv
        fhs = []
        try:
            model_writer = {}
            for model in self.models:
                fhs.append(open(model, 'w+', newline='', encoding='utf-8-sig'))
                model_writer[model] = csv.writer(fhs[-1])
                model_writer[model].writerow(
                    ["融合节点名", "融合类型", "计算指令统计", "待支持的计算指令", "符号表达式",
                     "Loop表达", "MTE优化统计", "Loop大小统计", "符号细节统计"])
            yield model_writer
        finally:
            for fh in fhs:
                fh.close()

    def save_csv(self, fn: str):
        assert fn.endswith(".csv")
        self.models.add(fn)
        with self.open_model_csv_writers() as writers:
            writer = writers[fn]
            for name, untyped_sub in self.subs.items():
                sub: OpSummary = untyped_sub
                graph = _GraphVisitor(sub.graph)
                mte_ops = graph.load_ops + graph.store_ops

                used_exps = set()
                size_exp = [op.output_size() + op.output_stride() for op in mte_ops]
                for expr in [v for k in size_exp for v in k]:
                    used_exps.add(str(expr))
                exps = '\n'.join(set(used_exps))
                ops_count = '\n'.join([f"{k}: {v}" for k, v in graph.typed_ops_count.items()])
                unsupported_ops = '\n'.join([f"{v}" for v in sub.graph.unsupported_ops])

                sorted_calls = sorted(sub.calls.items(), key=lambda x: x[1], reverse=True)
                sorted_calls_str = '\n'.join([f"{item[1]}次：{item[0]}" for item in sorted_calls])

                optimized_detail = f"mte2减少: {graph.num_load_optimized}个\n"
                optimized_detail += f"mte3减少: {graph.num_store_optimized}个"

                sorted_calls_detail = sorted(sub.calls_detail.items(), key=lambda x: x[1], reverse=True)
                sorted_calls_detail_str = '\n'.join([f"{item[1]}次：{item[0]}" for item in sorted_calls_detail])

                row = [name, graph.graph_type, ops_count, unsupported_ops, exps, sub.loop, optimized_detail,
                       sorted_calls_str, sorted_calls_detail_str]
                writer.writerow(row)
                if sub.model_path and sub.model_path in writers:
                    writers[sub.model_path].writerow(row)

    def __str__(self):
        for _, sub in self.subs.items():
            self.num_supported += sub.num_supported
            self.num_unsupported += sub.num_unsupported
            for op_type, count in sub.supported.items():
                self.supported[op_type] += count
            for op_type, count in sub.unsupported.items():
                self.unsupported[op_type] += count
        summary = f"Summary %{self.name}%:\n"
        summary += f"Supported op nums: {self.num_supported}/{self.num_unsupported + self.num_supported}\n"
        summary += f"Supported op type: {len(self.supported)}/{len(self.unsupported) + len(self.supported)}\n"
        summary += f"Supported ops:\n"
        sorted_supported = sorted(self.supported.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_supported:
            summary += f"    {op_type}: {count}\n"
        summary += f"Unsupported ops:\n"
        sorted_unsupported = sorted(self.unsupported.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_unsupported:
            summary += f"    {op_type}: {count}\n"
        summary += f"Fallback aten ops:\n"
        sorted_fallbacks = sorted(self.fallbacks.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_fallbacks:
            summary += f"    {op_type}: {count}, reason: {self.fallback_reasons[op_type]}\n"
        return summary


OP_SUMMARY = OpSummary("Model")
atexit.register(lambda: OP_SUMMARY.save())


def _left_align_lines(lines: List[str]):
    max_len = max([len(l) for l in lines])
    for i, l in enumerate(lines):
        lines[i] = l.ljust(max_len)
    return lines


def _value_str(value):
    if isinstance(value, (list, tuple)):
        value = [StrRep(str(v)) for v in value]
    return str(value)


def _get_node_style(node: _Op):
    style = {
        "shape": "record",
        "fillcolor": "LightGreen",
        "style": '"filled,rounded"',
    }

    if node.op_type in ["Data", "Output"]:
        style["fillcolor"] = "AliceBlue"
    elif node.get_private_attr("compute_type") == 'reduce':
        style["fillcolor"] = "Bisque"
    elif node.op_type == "Broadcast":
        style["fillcolor"] = "LightBlue"
    elif node.op_type == "Transpose":
        style["fillcolor"] = "Yellow"
    elif node.op_type in ["Load", "Store"] and node.get_private_attr("not_contiguous"):
        style["fillcolor"] = "Pink"
    return style


def make_graph_dot(asc_graph: ASCGraph, *, parent=None, with_priavte_attrs=False):
    import pydot
    graph: pydot.Dot = pydot.Dot(rankdir="TB") if parent is None else parent
    axis_str = ', '.join([f"{k}:{v}" for k, v in asc_graph.axis_vars.items()])
    label = f"{asc_graph.name}({axis_str})"
    cluster = pydot.Cluster(asc_graph.name, label=label, labeljust='c')
    graph.add_subgraph(cluster)
    for untyped_op in asc_graph.ops:
        n: _Op = untyped_op
        label = "{"
        label += f"name={asc_graph.name}/{n.name}|type={n.op_type}"
        inputs = []
        for attr, value in n.attrs.items():
            if isinstance(value, _Tensor):
                inputs.append((attr, value.op.name))
            elif isinstance(value, (list, tuple)) and all(isinstance(v, _Tensor) for v in value):
                inputs.append((attr, [v.op.name for v in value]))
            else:
                attr = attr.replace(f"{n.name}.attr.", '')
                attr = attr.replace(f"{n.name}.", '')
                label += f"|{attr}={_value_str(value)}"

        for attr, value in n.private_attrs.items():
            if not with_priavte_attrs:
                continue
            label += f"|private.{attr}={_value_str(value)}"
        label += "}"

        dot_node = pydot.Node(f'{asc_graph.name}/{n.name}', label=label, **_get_node_style(n))
        for name, src in inputs:
            if isinstance(src, str):
                graph.add_edge(pydot.Edge(f'{asc_graph.name}/{src}',
                               f'{asc_graph.name}/{n.name}', label=f'{asc_graph.name}/{name}'))
                continue
            for i, s in enumerate(src):
                graph.add_edge(pydot.Edge(f'{asc_graph.name}/{s}', f'{asc_graph.name}/{n.name}',
                                          label=f'{asc_graph.name}/{name}[{i}]'))
        cluster.add_node(dot_node)
    return graph


def make_fused_graph_dot(fused_graph: FusedASCGraph, *, with_priavte_attrs=False):
    import pydot
    graph: pydot.Dot = pydot.Dot(rankdir="TB")

    buffer_readers: Dict[str, List[str]] = {}
    buffer_writers: Dict[str, List[str]] = {}

    input_cluster = pydot.Cluster("inputs", label="inputs", labeljust='c')
    cluster_node_style = {"shape": "record", "fillcolor": "AliceBlue", "style": '"filled,rounded"'}
    graph.add_subgraph(input_cluster)
    for i, read in enumerate(fused_graph.inputs):
        label = f"{{name={read}|index={i}}}"
        input_cluster.add_node(pydot.Node(read, label=label, **cluster_node_style))
        buffer_writers[read] = [read]

    for asc_graph in fused_graph.subgraphs:
        make_graph_dot(asc_graph, with_priavte_attrs=with_priavte_attrs, parent=graph)
        for read in asc_graph.inputs:
            buffer_readers.setdefault(read, [])
            buffer_readers[read].append(f'{asc_graph.name}/{read}')
        for write in asc_graph.outputs:
            buffer_writers.setdefault(write, [])
            buffer_writers[write].append(f'{asc_graph.name}/{write}')

    output_cluster = pydot.Cluster("outputs", label="outputs", labeljust='c')
    graph.add_subgraph(output_cluster)
    for i, output in enumerate(fused_graph.outputs):
        label = f"{{name={output}|index={i}}}"
        output_cluster.add_node(pydot.Node(output, label=label, **cluster_node_style))
        for writer in buffer_writers.get(output, []):
            graph.add_edge(pydot.Edge(writer, output))

    for buffer, readers in buffer_readers.items():
        for reader in readers:
            for writer in buffer_writers.get(buffer, []):
                graph.add_edge(pydot.Edge(writer, reader))

    return graph


def save_asserts(name, context, fn):
    from torch._inductor import config
    if not config.trace.enabled:
        return

    from torch._inductor.virtualized import V

    asserts_dir = V.debug.filename(name)
    os.makedirs(asserts_dir, exist_ok=True)

    fd = os.open(os.path.join(asserts_dir, fn), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, 'w+') as f:
        f.write(context)
