import torch  # for globals


class VType:
    def __getattr__(self, item):
        return item


class VAttr(object):
    def __init__(self, name, parent):
        self._v = None
        self.attrs = {}
        self.name = f"{parent}.{name}" if parent else name

    def __getattr__(self, item):
        if item not in self.attrs:
            self.attrs[item] = VAttr(item, self.name)
        return self.attrs[item]

    def reals(self):
        real_attrs = []
        for attr in self.attrs.values():
            real_attrs.extend([attr] if attr.v else attr.reals())
        return real_attrs

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        self._v = v


class VOpIn(VAttr):
    def __init__(self, op, name: str):
        super().__init__(name, op.name)
        self.op = op
        self.name = name
        self.index = int(name[1:] if name[1:] else 0)
        self.src = None

    @property
    def v(self):
        return self

    @v.setter
    def v(self, v):
        assert isinstance(v, VOpOut)
        assert v.op.name != self.op.name, f"Cycle detected in {self.op.name}->{v.op.name}"
        self.src = v


class VOpOut(VAttr):
    def __init__(self, op, name: str):
        super().__init__(name, op.name)
        self.op = op
        self.name = name
        self.index = int(name[1:] if name[1:] else 0)

    def __str__(self):
        return f"{self.op.name}:{self.index}"


class VOp(object):
    def __init__(self, op):
        self.type = op
        self.name = None
        self._attrs = {}
        self.inputs = {}
        self.outputs = {}

    @property
    def attrs(self):
        attrs = []
        for attr in self._attrs.values():
            if attr.v:
                attrs.append(attr)
            else:
                attrs.extend(attr.reals())
        return attrs

    def __call__(self, name):
        self.name = name
        return self

    def __getattr__(self, item: str):
        if item.startswith("x"):
            if item not in self.inputs:
                self.inputs[item] = VOpIn(self, item)
            return self.inputs[item]

        if item.startswith("y"):
            if item not in self.outputs:
                self.outputs[item] = VOpOut(self, item)
            return self.outputs[item]

        if item not in self._attrs:
            self._attrs[item] = VAttr(item, self.name)
        return self._attrs[item]

    def __str__(self):
        return f"{self.name}({self.type}){[str(v) for v in self._attrs.values()]}"


class VOps:
    def __init__(self, graph):
        self.graph = graph

    def __getattr__(self, item):
        self.graph.nodes.append(VOp(item))
        return self.graph.nodes[-1]


class VGraph:
    def __init__(self, name):
        self.name = name
        self.OP = VOps(self)
        self.nodes = []
        self.inputs = []
        self.outputs = []

    @property
    def ops(self):
        return self.OP

    def create_size(self, name):
        return name

    def create_axis(self, name, range):
        return name

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_outputs(self, outputs):
        self.outputs = outputs

    def expr(self, sizes):
        if len(sizes) == 0:
            return "1"
        return "*".join([str(s) for s in sizes])


class VASCGraph:
    def __init__(self, code):
        self.code = None
        self.call = None
        self.style_template = {
            "shape": "record",
            "fillcolor": "#CAFFE3",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
        }

        lines = []
        graph_start = False
        for i, line in enumerate(code.splitlines()):
            if "def" in line:
                self.call = line.split("def")[1].strip().split(":")[0]
                lines.append(line)
                continue
            if "HintGraph" in line:
                lines.append(line.replace("ascir.HintGraph", "VGraph"))
                graph_start = True
                continue
            if not graph_start:
                continue
            line = line.replace("ascir.ops", "graph.ops")
            line = line.replace("ascir.SizeExpr", "graph.expr")
            line = line.replace("ascir.dtypes", "VType()")

            def trigger_setv(match):
                return f"{match.group(1)}.v = {match.group(2)}"

            import re
            pattern = r'(.*\.\w+)\s*=\s*(.*)'
            line = re.sub(pattern, trigger_setv, line)

            lines.append(line)
        lines.append(f"graph = {self.call}")
        self.code = "\n".join(lines)
        local_vars = {}
        exec(self.code, globals(), local_vars)
        self.graph: VGraph = local_vars["graph"]

    def debug_str(self):
        from torch._inductor.codegen.common import IndentedBuffer
        s = IndentedBuffer()
        for node in self.graph.nodes:
            s.writeline(f"{node.name, node.type}")
            for input in node.inputs.values():
                with s.indent():
                    s.writeline(f"Input{input.index}: {input.src}")
                    with s.indent():
                        for attr in input.reals():
                            s.writeline(f"{attr.name}: {attr.v}")
                        for attr in input.src.reals():
                            s.writeline(f"{attr.name}: {attr.v}")
            with s.indent():
                for attr in node.attrs:
                    s.writeline(f"{attr.name}: {attr.v}")
        return s.getvalue()

    def view_dot(self):
        try:
            import pydot
        except ImportError:
            print("Please install pydot first.")
            return
        graph: pydot.Dot = pydot.Dot(rankdir="TB")
        cluster_nodes = []
        type_colors = {"Data": "AliceBlue", "Broadcast": "LightBlue"}
        for n in self.graph.nodes:
            n: VOp = n
            label = "{"
            label += f"name={n.name}|type={n.type}|"
            attr_prefix = f"{n.name}.attr."
            label += "|".join([f"{attr.name.replace(attr_prefix, '')}={attr.v}" for attr in n.attrs])
            for output in n.outputs.values():
                for attr in output.reals():
                    label += f"|{attr.name}={attr.v}"
            label += "}"
            style = self.style_template.copy()
            if n.type in type_colors:
                style["fillcolor"] = type_colors[n.type]
            dot_node = pydot.Node(n.name, label=label, **style)
            for i, input in n.inputs.items():
                graph.add_edge(pydot.Edge(input.src.op.name, n.name, label=str(i), fontsize='8'))

            if n.type != "Data":
                cluster_nodes.append(dot_node)
                continue

            cluster = pydot.Cluster(n.name, label=n.name, labeljust='c')
            cluster.add_node(dot_node)
            if n.name.startswith("buf"):
                for node in cluster_nodes:
                    cluster.add_node(node)
                cluster_nodes.clear()
            graph.add_subgraph(cluster)

        graph.write_svg(f"./{self.graph.name}.svg")


def draw_asc_graph_dot(body: str):
    from torch._inductor.codegen.common import IndentedBuffer
    graph = IndentedBuffer()
    graph.writeline("def build():")
    with graph.indent():
        graph.splice(body)
    graph.writeline("graph = build()")
    v_graph = VASCGraph(graph.getvalue())
    v_graph.view_dot()