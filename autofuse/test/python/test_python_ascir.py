import pytest
from pyautofuse import ascir, Autofuser


class TestAscir():
    @staticmethod
    def test_graph_create_size():
        graph = ascir.HintGraph("test")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        graph.set_inputs([])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "Nodes:\n",
        ])

    @staticmethod
    def test_graph_create_axis():
        graph = ascir.HintGraph("test")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", ascir.SizeExpr([s0]))
        z1 = graph.create_axis("z1", ascir.SizeExpr([s1]))
        z2 = graph.create_axis("z2", ascir.SizeExpr([s2]))

        graph.set_inputs([])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0: s0, ORIGINAL\n",
            "  z1: s1, ORIGINAL\n",
            "  z2: s2, ORIGINAL\n",
            "Nodes:\n",
        ])

    @staticmethod
    def test_graph_create_node():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x")

        graph.set_inputs([x])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
        ])

    @staticmethod
    def test_graph_create_node_with_attr():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x")
        cast = ascir.ops.Cast("cast")
        cast.x = x
        cast.dst_type = 10
        cast.attr.hint.compute_type = "elemwise"

        graph.set_inputs([x])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "  cast: Cast (0)\n",
            "    .dst_type = 10\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = elewise\n"
            "    .x = x.y\n"
            "    .y.dtype = float32\n",
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
        ])

    @staticmethod
    def test_graph_create_node_with_axis():
        graph = ascir.HintGraph("test")

        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", ascir.SizeExpr([s0]))
        z1 = graph.create_axis("z1", ascir.SizeExpr([s1]))
        z2 = graph.create_axis("z2", ascir.SizeExpr([s2]))

        x = ascir.ops.Data("x")
        x.attr.sched.exec_order = 0
        x.attr.sched.axis = [z0, z1, z2]
        x.y.dtype = ascir.dtypes.float16
        x.y.axis = [z0, z1, z2]
        x.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        x.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        graph.set_inputs([x])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0: s0, ORIGINAL\n",
            "  z1: s1, ORIGINAL\n",
            "  z2: s2, ORIGINAL\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float16\n",
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
         ])

    @staticmethod
    def test_graph_link_nodes():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x")

        load = ascir.ops.Load("load")
        load.x = x

        graph.set_inputs([x])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "  load: Load (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = x.y\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
        ])

    @staticmethod
    def test_graph_link_nodes_by_output():
        graph = ascir.HintGraph("test")

        x = ascir.ops.Data("x")

        load = ascir.ops.Load("load")
        load.x = x.y

        graph.set_inputs([x])
        graph.set_outputs([])

        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: test\n",
            "Sizes:\n",
            "Axis:\n",
            "Nodes:\n",
            "  x: Data (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n",
            "    .y.vectorized_axis = {}\n"
            "  load: Load (0)\n",
            "    .axis = {}\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = x.y\n"
            "    .y.dtype = float32\n"
            "    .y.axis = {}\n",
            "    .y.repeats = {}\n",
            "    .y.strides = {}\n"
            "    .y.vectorized_axis = {}\n"
        ])


class TestAutofuseLoadAbsStore():
    @staticmethod
    def construct_graph():
        graph = ascir.HintGraph("LoadAbsStore")
        s0 = graph.create_size("s0")
        s1 = graph.create_size("s1")
        s2 = graph.create_size("s2")

        z0 = graph.create_axis("z0", ascir.SizeExpr([s0]))
        z1 = graph.create_axis("z1", ascir.SizeExpr([s1]))
        z2 = graph.create_axis("z2", ascir.SizeExpr([s2]))

        arg3_1 = ascir.ops.Data("arg3_1")
        arg3_1.attr.sched.exec_order = 0
        arg3_1.attr.sched.axis = [z0, z1, z2]
        arg3_1.y.dtype = ascir.dtypes.float16
        arg3_1.y.axis = [z0, z1, z2]
        arg3_1.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        arg3_1.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        load = ascir.ops.Load("load")
        load.x = arg3_1
        load.attr.sched.exec_order = 1
        load.attr.sched.axis = [z0, z1, z2]
        load.y.dtype = ascir.dtypes.float16
        load.y.axis = [z0, z1, z2]
        load.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        load.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        abs_op = ascir.ops.Abs("abs")
        abs_op.x = load
        abs_op.attr.sched.exec_order = 2
        abs_op.attr.sched.axis = [z0, z1, z2]
        abs_op.y.dtype = ascir.dtypes.float16
        abs_op.y.axis = [z0, z1, z2]
        abs_op.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        abs_op.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        store = ascir.ops.Store("store")
        store.x = abs_op
        store.attr.sched.exec_order = 3
        store.attr.sched.axis = [z0, z1, z2]
        store.y.dtype = ascir.dtypes.float16
        store.y.axis = [z0, z1, z2]
        store.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        store.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        buf1 = ascir.ops.Data("buf1")
        buf1.x = store
        buf1.attr.sched.exec_order = 4
        buf1.attr.sched.axis = [z0, z1, z2]
        buf1.y.dtype = ascir.dtypes.float16
        buf1.y.axis = [z0, z1, z2]
        buf1.y.size = [ascir.SizeExpr([s0]), ascir.SizeExpr([s1]), ascir.SizeExpr([s2])]
        buf1.y.strides = [ascir.SizeExpr([s1, s2]), ascir.SizeExpr([s2]), ascir.SizeExpr()]

        graph.set_inputs([arg3_1])
        graph.set_outputs([])

        return graph

    def test_construct_graph(self):
        graph = self.construct_graph()
        debug_str = ascir.utils.debug_str(graph)
        assert debug_str == "".join([
            "Graph: LoadAbsStore\n",
            "Sizes:\n",
            "  s0: VAR\n",
            "  s1: VAR\n",
            "  s2: VAR\n",
            "Axis:\n",
            "  z0: s0, ORIGINAL\n",
            "  z1: s1, ORIGINAL\n",
            "  z2: s2, ORIGINAL\n",
            "Nodes:\n",
            "  arg3_1: Data (0)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = nil\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "  load: Load (1)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = arg3_1.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"
            "  abs: Abs (2)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = load.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"            
            "  store: Store (3)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = abs.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"            
            "  buf1: Data (4)\n",
            "    .axis = {z0, z1, z2, }\n",
            "    .hint:\n"
            "      .compute_type = data\n"
            "    .x = store.y\n"
            "    .y.dtype = float16\n"
            "    .y.axis = {z0, z1, z2, }\n",
            "    .y.repeats = {s0, s1, s2, }\n",
            "    .y.strides = {s1*s2, s2, 1, }\n",
            "    .y.vectorized_axis = {}\n"            
        ])

    def test_optimize(self):
        options = {}
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.autofuse(hint_graph)

    @pytest.mark.skip
    def test_codegen(self):
        options = {}
        fuser = Autofuser(options)

        hint_graph = self.construct_graph()
        impl_graphs = fuser.autofuse(hint_graph)
        op_proto, tiling_def, host_tiling, op_kernel = fuser.codegen(impl_graph)
