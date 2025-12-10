import importlib
import pickle
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from packaging import version

import torch
from torch._guards import TracingContext
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_aggregate
from torch.utils._python_dispatch import _disable_current_modes

if version.parse(torch.__version__) < version.parse("2.5.1"):
    # compile_lock is a threading.RLock() that was introduced in version 2.5.1.
    compile_lock = threading.RLock()
else:
    from torch._dynamo.convert_frame import compile_lock


class _ArgWrapper:
    """Helper function for storing fx.Node arg that are nodes"""

    def __init__(self, n: Node):
        self.name = n.name


def _get_target_function(fn_name: str):
    fqn_list = fn_name.split(".")
    try:
        target = importlib.import_module(fqn_list[0])
        for attr in fqn_list[1:]:
            target = getattr(target, attr)
        if not callable(target):
            raise TypeError(f"Target function must be Callable.")
    except Exception as e:
        raise NotImplementedError(f"Unsupported call_function: {fn_name} with error msg: {str(e)}.") from e
    return target


class SerializableArtifact(ABC):
    def __init__(self, artifact):
        if isinstance(artifact, SerializableArtifact):
            return
        super().__init__()
        if not callable(artifact):
            raise TypeError(f"artifact must be Callable.")
        self._artifact = artifact
        if getattr(artifact, "_boxed_call", False):
            self._boxed_call = True

    def __call__(self, *args, **kwargs):
        return self._artifact(*args, **kwargs)

    def __new__(cls, artifact):
        if isinstance(artifact, SerializableArtifact):
            return artifact
        else:
            return super().__new__(cls)

    @property
    def artifact(self):
        return self._artifact

    def __reduce__(self):
        return self.rebuild_from_bytes, (self.convert_to_bytes(),)

    @abstractmethod
    def convert_to_bytes(self) -> bytes:
        """
        Convert artifact to bytes. The return value should be artifact_bytes,
        which can rebuild the artifact via rebuild_from_bytes.
        """
        ...

    @staticmethod
    @abstractmethod
    def rebuild_from_bytes(byte_s: bytes):
        """
        Rebuild artifact from bytes. The input is the artifact_bytes from convert_to_bytes.
        """
        ...


@contextmanager
def temp_disable_tracing_envs():
    with compile_lock, _disable_current_modes():
        if version.parse(torch.__version__) < version.parse("2.5.1"):
            tracing_context = TracingContext.get()
            if tracing_context is not None:
                tracing_context_fake_mode_old = tracing_context.fake_mode
                tracing_context.fake_mode = None
                try:
                    yield
                finally:
                    tracing_context.fake_mode = tracing_context_fake_mode_old
            else:
                yield
        else:
            tracing_context = TracingContext.try_get()
            with tracing_context.patch(fake_mode=None) if tracing_context else nullcontext():
                yield


class SerializableGraphModule(SerializableArtifact):
    def __init__(self, artifact):
        if not isinstance(artifact, GraphModule):
            raise TypeError(f"artifact must be GraphModule, got {type(artifact)}")
        super().__init__(artifact)

    def convert_to_bytes(self) -> bytes:
        with temp_disable_tracing_envs():
            gm_dict, graph_meta, nodes_meta = GmSerializeHelper.serialize_fn(self._artifact)
            return pickle.dumps((gm_dict, graph_meta, nodes_meta))

    @staticmethod
    def rebuild_from_bytes(byte_s: bytes):
        with temp_disable_tracing_envs():
            gm_dict, graph_meta, nodes_meta = pickle.loads(byte_s)
            gm = GmSerializeHelper.deserialize_fn(gm_dict, graph_meta, nodes_meta)
            return __class__(gm)


class GmSerializeHelper:
    """Note: this is a backported serializer / deserializer for class GraphModule"""

    @staticmethod
    def serialize_fn(gm: GraphModule):
        gm_dict = gm.__dict__.copy()
        del gm_dict["_graph"]
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, GraphModule):
                gm_dict["_modules"][k] = GmSerializeHelper.serialize_fn(v)
        graph = gm.graph
        graph_meta = (graph._tracer_cls, graph._tracer_extras)
        nodes = list(graph.nodes)
        nodes_meta = []

        def _wrap_arg(arg):
            if isinstance(arg, Node):
                return _ArgWrapper(arg)
            else:
                return arg

        for node in nodes:
            node_meta = (
                node.name,
                node.type,
                node.op,
                node._pretty_print_target(node.target),
                tuple(map_aggregate(node.args, _wrap_arg)),
                dict(map_aggregate(node.kwargs, _wrap_arg)),
            )
            nodes_meta.append(node_meta)

        return gm_dict, graph_meta, nodes_meta

    @staticmethod
    def deserialize_fn(gm_dict, graph_meta, nodes_meta):
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, tuple):
                gm_dict["_modules"][k] = GmSerializeHelper.deserialize_fn(*v)
        gm = GraphModule.__new__(GraphModule)
        gm.__dict__ = gm_dict

        tracer_cls, tracer_extras = graph_meta
        graph = Graph(gm, tracer_cls, tracer_extras)

        _node_name_to_node = {}

        def _unwrap_arg(arg):
            if isinstance(arg, _ArgWrapper):
                return _node_name_to_node[arg.name]
            else:
                return arg

        for node_meta in nodes_meta:
            node_name, node_type, node_op, node_target, node_args, node_kwargs = node_meta

            if node_op == "call_function":
                node_target = _get_target_function(node_target)

            node_args = tuple(map_aggregate(node_args, _unwrap_arg))
            node_kwargs = dict(map_aggregate(node_kwargs, _unwrap_arg))
            _node_name_to_node[node_name] = graph.create_node(
                node_op, node_target, node_args, node_kwargs, node_name, node_type
            )
        gm.graph = graph
        gm.recompile()
        return gm