import atexit
import operator
from collections import defaultdict
from typing import Iterable, Dict, List

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
    def __init__(self, graph, *, loop=""):
        self.graph = ASCGraph(name=graph) if isinstance(graph, str) else graph
        self.loop = loop
        self.name = self.graph.name
        self.fallbacks: Dict[str:int] = defaultdict(lambda: 0)
        self.fallback_reasons: Dict[str:int] = defaultdict(lambda: "unknown")
        self.supported: Dict[str:int] = defaultdict(lambda: 0)
        self.unsupported: Dict[str:int] = defaultdict(lambda: 0)
        self.subs: Dict[str:OpSummary] = {}
        self.num_supported: int = 0
        self.num_unsupported: int = 0
        for op in self.graph.ops:
            self.record(op.op_type, f"{op.name}.attr.is_unsupported" not in op.attrs)

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

    def add_graph_summary(self, graph, *, loop):
        if graph.name not in self.subs:
            self.subs[graph.name] = OpSummary(graph, loop=loop)
        return self.subs[graph.name]

    def save(self):
        self.save_csv(f"{self.name}.csv")
        print(str(self))

    def save_csv(self, fn: str):
        import csv
        assert fn.endswith(".csv")
        with open(fn, 'w+', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            print(f"Save summary to {fn}")
            writer.writerow(["融合节点名", "融合类型", "不支持的操作符", "不支持的表达式", "计算指令统计", "Loop表达"])
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

                unsupported_exprs = set()
                unsupported_operators = set()
                supported_operators = [operator.mul]
                size_exp = [op.output_size() + op.output_stride() for op in mte_ops]
                for expr in [v for k in size_exp for v in k]:
                    used_operators = [v for v in expr.asc_expr.operators(True) if v not in supported_operators]
                    if len(used_operators):
                        unsupported_operators.update(used_operators)
                        unsupported_exprs.add(str(expr.expr))
                unsupported_exprs = '\n'.join(set(unsupported_exprs))
                typed_ops_count = '\n'.join([f"{k}: {v}" for k, v in typed_ops_count.items()])
                unsupported_operators = '\n'.join(
                    [v.__name__ if hasattr(v, '__name__') else str(v) for v in unsupported_operators])
                writer.writerow([name, graph_type, unsupported_operators, unsupported_exprs, typed_ops_count, sub.loop])

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
    cluster_nodes = []
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
        for attr, value in n.attrs.items():
            if isinstance(value, _Tensor):
                inputs.append((attr, value.op.name))
            else:
                attr = attr.replace(f"{n.name}.attr.", '')
                attr = attr.replace(f"{n.name}.", '')
                if attr == "is_unsupported":
                    style["fillcolor"] = "#FF0000"
                    continue
                if isinstance(value, (list, tuple)):
                    value = [StrRep(str(v)) for v in value]
                label += f"|{attr}={str(value)}"
        label += "}"

        if n.op_type in type_colors:
            style["fillcolor"] = type_colors[n.op_type]
        dot_node = pydot.Node(n.name, label=label, **style)
        for name, src in inputs:
            graph.add_edge(pydot.Edge(src, n.name, label=str(name)))

        if n.op_type not in ["Data", "Output", "Workspace"]:
            cluster_nodes.append(dot_node)
            continue

        cluster = pydot.Cluster(n.name, label=n.name, labeljust='c')
        cluster.add_node(dot_node)
        if n.name.startswith("buf"):
            for node in cluster_nodes:
                cluster.add_node(node)
            cluster_nodes.clear()
        graph.add_subgraph(cluster)

    return graph


def draw_asc_graph_dot(asc_graph: ASCGraph, file_path=None):
    graph = make_graph_dot(asc_graph)
    file_path = file_path if file_path else f"./{asc_graph.name}.svg"
    graph.write_svg(file_path)
