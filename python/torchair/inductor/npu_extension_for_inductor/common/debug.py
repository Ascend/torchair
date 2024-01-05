from collections import defaultdict
from typing import Iterable, Dict

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.symbols import AscSymbol
from npu_extension_for_inductor.common.utils import StrRep
from npu_extension_for_inductor.ir import _Op, _Tensor


class OpSummary:
    def __init__(self, name):
        self.name = name
        self.supported: Dict[str:int] = defaultdict(lambda: 0)
        self.unsupported: Dict[str:int] = defaultdict(lambda: 0)
        self.subs: Dict[str:OpSummary] = {}
        self.num_supported: int = 0
        self.num_unsupported: int = 0

    def record(self, op, is_supported):
        if is_supported:
            self.num_supported += 1
            self.supported[op] += 1
        else:
            self.num_unsupported += 1
            self.unsupported[op] += 1

    def add_graph_summary(self, name):
        if name not in self.subs:
            self.subs[name] = OpSummary(name)
        return self.subs[name]

    def __str__(self):
        for name, sub in self.subs.items():
            self.num_supported += sub.num_supported
            self.num_unsupported += sub.num_unsupported
            for op_type, count in sub.supported.items():
                self.supported[op_type] += count
            for op_type, count in sub.unsupported.items():
                self.unsupported[op_type] += count
        summary = f"Summary {self.name}:\n"
        summary += f"  Supported op type:{len(self.supported)}/{len(self.unsupported) + len(self.supported)}\n"
        summary += f"  Supported op nums:{self.num_supported}/{self.num_unsupported + self.num_supported}\n"
        summary += f"  Unsupported op type: {self.num_unsupported}\n"
        sorted_unsupported = sorted(self.unsupported.items(), key=lambda x: x[1], reverse=True)
        for op_type, count in sorted_unsupported:
            summary += f"    {op_type}: {count}\n"
        return summary


OP_SUMMARY = OpSummary("Model")


def draw_asc_graph_dot(asc_graph: ASCGraph, file_path=None):
    try:
        import pydot
    except ImportError:
        print("Please install pydot first.")
        return
    graph: pydot.Dot = pydot.Dot(rankdir="TB")
    cluster_nodes = []
    type_colors = {"Data": "AliceBlue", "Broadcast": "LightBlue"}
    summary = OP_SUMMARY.add_graph_summary(asc_graph.name)
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
        is_supported = True
        for attr, value in n.attrs.items():
            if isinstance(value, _Tensor):
                inputs.append((attr, value.op.name))
            else:
                attr = attr.replace(f"{n.name}.attr.", '')
                attr = attr.replace(f"{n.name}.", '')
                if attr == "is_unsupported":
                    is_supported = False
                    style["fillcolor"] = "#FF0000"
                    continue
                if isinstance(value, (list, tuple)):
                    value = [StrRep(str(v)) for v in value]
                label += f"|{attr}={str(value)}"
        label += "}"

        if n.op_type in type_colors:
            style["fillcolor"] = type_colors[n.op_type]
        dot_node = pydot.Node(n.name, label=label, **style)
        summary.record(n.op_type, is_supported)
        for name, src in inputs:
            graph.add_edge(pydot.Edge(src, n.name, label=str(name)))

        if n.op_type != "Data":
            cluster_nodes.append(dot_node)
            continue

        cluster = pydot.Cluster(n.name, label=n.name, labeljust='c')
        cluster.add_node(dot_node)
        if n.name.startswith("buf"):
            for node in cluster_nodes:
                cluster.add_node(node)
            cluster_nodes.clear()
        graph.add_subgraph(cluster)

    graph.add_node(pydot.Node("_summary", label=str(summary), shape="plaintext"))

    file_path = file_path if file_path else f"./{asc_graph.name}.svg"
    graph.write_svg(file_path)