from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools

import torch
from torch import Tensor
from torch.fx.node import Argument, Target

from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack
from torchair.configs.compiler_config import CompilerConfig

import tensorflow
tf = tensorflow.compat.v1
tf.disable_v2_behavior()


__CONVERTERS = defaultdict(None)


def _get_converter(name: Callable):
    global __CONVERTERS
    return __CONVERTERS[name]


def register_fx_node_tf_converter(aten_op, converter: Callable = None):
    if converter is not None:
        global __CONVERTERS
        __CONVERTERS.update({aten_op: converter})
        return converter

    def register_demo(f, key):
        global __CONVERTERS
        __CONVERTERS.update({key: f})
        return f

    return functools.partial(register_demo, key=aten_op)


def _torch_type_to_tf_type(dtype):
    if dtype == torch.float32:
        return tf.float32
    elif dtype == torch.int32:
        return tf.int32
    elif dtype == torch.bool:
        return tf.bool
    else:
        raise RuntimeError(f"Unsupported torch type {dtype} by tf")


class TfConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, graph=None):
        self._graph = tf.Graph() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []
        self._config = config

    def context(self):
        return self._graph.as_default()

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if isinstance(meta_outputs, torch.SymInt):
            self._inputs.append(tf.placeholder(name=target, dtype=tf.int32))
        else:
            assert isinstance(meta_outputs, torch.Tensor)
            self._inputs.append(tf.placeholder(
                name=target, dtype=_torch_type_to_tf_type(meta_outputs.dtype)))
        return self._inputs[-1]

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        assert isinstance(args, (list, tuple)) and len(args) == 1
        args = args[0]
        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            if isinstance(arg, tf.Tensor):
                self._fx_outputs_mapping[len(
                    self._fx_outputs)] = len(self._outputs)
                self._outputs.append(arg)
            self._fx_outputs.append(arg)

        return args

    def parse_symlist(self, syms):
        concat_inputs = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                concat_inputs.append(sym.npu)
            else:
                assert isinstance(sym, int)
                concat_inputs.append(tf.constant(sym, name=f"{sym}"))

        return tf.stack(concat_inputs, axis=0)

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        converter = _get_converter(target)
        if converter is None:
            raise RuntimeError(f"Unsupported torch op {target} by tf")
        return converter(args, kwargs, meta_outputs)

    def dump(self, path: str):
        if path is None:
            return
        with open(path, "w+") as f:
            f.write(str(self.graph.as_graph_def()))

    def compile(self) -> Any:
        pass

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with tf.Session(graph=self.graph) as sess:
            args = [arg.numpy() if isinstance(arg, torch.Tensor)
                    else arg for arg in args]
            tf_outputs = sess.run(
                self.outputs, feed_dict=dict(zip(self.inputs, args)))
            for k, v in self._fx_outputs_mapping.items():
                self._fx_outputs[k] = torch.Tensor(tf_outputs[v])
            return tuple(self._fx_outputs)


@register_fx_node_tf_converter(torch.ops.aten.sym_size)
def conveter_sym_size(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    shape = tf.shape(args[0])
    return tf.gather(shape, args[1])


@register_fx_node_tf_converter(torch.ops.aten.cat.default)
def conveter_cat(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return tf.concat(args[0], 0 if len(args) == 1 else args[1])


@register_fx_node_tf_converter(torch.ops.aten.ones.default)
def conveter_ones(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return tf.ones(args[0])


@register_fx_node_tf_converter(torch.ops.aten.add.Tensor)
def conveter_add(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    return tf.add(args[0], args[1])


@register_fx_node_tf_converter(torch.ops.aten.split.Tensor)
def conveter_split(args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
    split_sizes = args[1]
    if isinstance(split_sizes, int):
        split_sizes = [args[1] for _ in range(len(meta_outputs))]
        split_sizes[-1] = -1
    return tf.split(args[0], split_sizes)
