from typing import Iterable

from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.symbols import AscSymbol
from npu_extension_for_inductor.common.utils import StrRep
from npu_extension_for_inductor.ir import _Op, _Tensor


def draw_asc_graph_dot(asc_graph: ASCGraph, file_path=None):
    try:
        import pydot
    except ImportError:
        print("Please install pydot first.")
        return
    graph: pydot.Dot = pydot.Dot(rankdir="TB")
    cluster_nodes = []
    type_colors = {"Data": "AliceBlue", "Broadcast": "LightBlue"}
    for untyped_op in asc_graph.ops:
        n: _Op = untyped_op
        label = "{"
        label += f"name={n.name}|type={n.op_type}"

        inputs = []
        for attr, value in n.attrs.items():
            if isinstance(value, _Tensor):
                inputs.append((attr, value.op.name))
            else:
                attr = attr.replace(f"{n.name}.attr.", '')
                attr = attr.replace(f"{n.name}.", '')
                if isinstance(value, (list, tuple)):
                    value = [StrRep(str(v)) for v in value]
                label += f"|{attr}={str(value)}"
        label += "}"

        style = {
            "shape": "record",
            "fillcolor": "#CAFFE3",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
        }
        if n.op_type in type_colors:
            style["fillcolor"] = type_colors[n.op_type]
        dot_node = pydot.Node(n.name, label=label, **style)
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

    file_path = file_path if file_path else f"./{asc_graph.name}.svg"
    graph.write_svg(file_path)