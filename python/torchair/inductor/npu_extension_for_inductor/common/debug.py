import atexit
import operator
from collections import defaultdict
import contextlib
from typing import Iterable, Dict, List, Set

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.symbols import AscSymbol, AscExpr
from npu_extension_for_inductor.common.utils import StrRep
from npu_extension_for_inductor.ir import _Op, _Tensor


class _OpVisitor:
    def __init__(self, op):
        self.op = op

    def output_size(self, name='y'):
        return self.op.attrs[f'{self.op.name}.{name}.size']

    def output_stride(self, name='y'):
        return self.op.attrs[f'{self.op.name}.{name}.strides']

    def input_size(self, name='x'):
        return self.op.attrs[f'{self.op.name}.{name}.size']

    def input_stride(self, name='x'):
        return self.op.attrs[f'{self.op.name}.{name}.strides']


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

    def save(self):
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
                    ["融合节点名", "融合类型", "计算指令统计", "待支持的计算指令", "待支持的操作符", "待支持的表达式",
                     "Loop表达"])
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
                mte_ops = [_OpVisitor(op) for op in sub.graph.ops if op.op_type in ["Load", "Store"]]
                workspace = [_OpVisitor(op) for op in sub.graph.ops if op.op_type == "Workspace"]
                reduction = [_OpVisitor(op) for op in sub.graph.ops if op.op_type == "Reduction"]
                graph_type = "Fused" if len(workspace) else "Reduction" if len(reduction) else "PointWise"

                typed_ops_count = {}
                for op in sub.graph.ops:
                    if op.op_type in ["Data", "Output", "Workspace"]:
                        continue
                    if op.op_type not in typed_ops_count:
                        typed_ops_count[op.op_type] = 0
                    typed_ops_count[op.op_type] += 1

                unsupported_exps = set()
                unsupported_operators = set()
                supported_operators = [operator.mul]
                size_exp = [op.output_size() + op.output_stride() for op in mte_ops]
                for expr in [v for k in size_exp for v in k]:
                    used_operators = [v for v in expr.asc_expr.operators(True) if v not in supported_operators]
                    if len(used_operators):
                        unsupported_operators.update(used_operators)
                        unsupported_exps.add(str(expr.expr))
                exps = '\n'.join(set(unsupported_exps))
                ops_count = '\n'.join([f"{k}: {v}" for k, v in typed_ops_count.items()])
                operators = '\n'.join(
                    [v.__name__ if hasattr(v, '__name__') else str(v) for v in unsupported_operators])
                unsupported_ops = '\n'.join([f"{v}" for v in sub.graph.unsupported_ops])
                row = [name, graph_type, ops_count, unsupported_ops, operators, exps, sub.loop]
                writer.writerow(row)
                if sub.model_path and sub.model_path in writers:
                    writers[sub.model_path].writerow(row)

    def __str__(self):
        for name, sub in self.subs.items():
            self.num_supported += sub.num_supported
            self.num_unsupported += sub.num_unsupported
            for op_type, count in sub.supported.items():
                self.supported[op_type] += count
            for op_type, count in sub.unsupported.items():
                self.unsupported[op_type] += count
        summary = f"Summary %{self.name}%:\n"
        summary += f"Supported op nums: {self.num_supported}/{self.num_unsupported + self.num_supported}\n"
        summary += f"Supported op type: {len(self.supported)}/{len(self.unsupported) + len(self.supported)}\n"
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


def _left_align_lines(lines: List[str]):
    max_len = max([len(l) for l in lines])
    for i, l in enumerate(lines):
        lines[i] = l.ljust(max_len)
    return lines


def _left_align_str(s: str):
    lines = s.split("\n")
    return "\n".join(_left_align_lines(lines))


def make_graph_dot(asc_graph: ASCGraph):
    try:
        import pydot
    except ImportError:
        print("Please install pydot first.")
        return
    graph: pydot.Dot = pydot.Dot(rankdir="TB")
    clusters: Dict[str, pydot.Cluster] = {}
    type_colors = {"Data": "AliceBlue", "Workspace": "Gray", "Output": "AliceBlue", "Broadcast": "LightBlue"}
    for untyped_op in asc_graph.ops:
        n: _Op = untyped_op
        style = {
            "shape": "record",
            "fillcolor": "#CAFFE3",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
        }

        label = "{"
        label += f"name={n.name}|type={n.op_type}"
        inputs = []
        if not n.supported:
            style["fillcolor"] = "#FF0000"
        for attr, value in n.attrs.items():
            if isinstance(value, _Tensor):
                inputs.append((attr, value.op.name))
            else:
                attr = attr.replace(f"{n.name}.attr.", '')
                attr = attr.replace(f"{n.name}.", '')
                if isinstance(value, (list, tuple)):
                    value = [StrRep(str(v)) for v in value]
                label += f"|{attr}={str(value)}"
        for attr, value in n.private_attrs.items():
            if isinstance(value, (list, tuple)):
                value = [StrRep(str(v)) for v in value]
            label += f"|private.{attr}={str(value)}"
        label += "}"

        if n.op_type in type_colors:
            style["fillcolor"] = type_colors[n.op_type]
        dot_node = pydot.Node(n.name, label=label, **style)
        for name, src in inputs:
            graph.add_edge(pydot.Edge(src, n.name, label=str(name)))

        buffer_name = n.get_private_attr("buffer_name")
        if buffer_name not in clusters:
            clusters[buffer_name] = pydot.Cluster(buffer_name, label=buffer_name, labeljust='c')
            graph.add_subgraph(clusters[buffer_name])

        cluster = clusters[buffer_name]
        cluster.add_node(dot_node)

    return graph


def draw_asc_graph_dot(asc_graph: ASCGraph, file_path=None):
    graph = make_graph_dot(asc_graph)
    file_path = file_path if file_path else f"./{asc_graph.name}.svg"
    graph.write_svg(file_path)
