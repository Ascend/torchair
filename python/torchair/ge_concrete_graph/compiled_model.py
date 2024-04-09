from typing import Any
from torchair.ge_concrete_graph.ge_graph import GeGraph
from torchair.core.backend import initialize_graph_engine, TorchNpuGraph
from torchair.core.utils import logger
from torchair.utils.export_utils import unserialize_dict_attr
from torchair.ge_concrete_graph.utils import InputProcessing
from torchair.ge_concrete_graph.ge_ir_pb2 import ModelDef


class CompiledModel(object):
    def __init__(self, graph: GeGraph = None):
        super(CompiledModel, self).__init__()
        self._graph = graph if graph is not None else GeGraph()
        self._input_process = None
        self._output_process = None
        self._fx_inputs_mapping = {}
        self._fx_outputs_mapping = {}
        self._graph_output_ref_input = {}
        self._local_compile_options = {}
        self._global_compile_options = {}
        self._executor = TorchNpuGraph(self._graph.name)
        self._is_compiled = False

    def __call__(self, *args: Any, **kwargs: Any):
        if not self._is_compiled:
            self.compile()

        inputs = self._input_process(*args)

        if len(self._graph_output_ref_input):
            assigned_outputs = [None] * len(self.graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in self._graph_output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]
            ge_outputs = self._executor.run(inputs, assigned_outputs)

        else:
            ge_outputs = self._executor.run(inputs)

        if len(ge_outputs) != len(self.graph.attr["_output_dtypes"].list.i):
            raise AssertionError(
                f"output size mismatch, expect {len(self.graph.attr['_output_dtypes'].list.i)}, got {len(ge_outputs)}")

        return ge_outputs

    def __str__(self) -> str:
        return str(self._graph)

    @property
    def graph(self):
        return self._graph

    @classmethod
    def from_serialized_str(cls, serialized):
        model_def = ModelDef()
        model_def.ParseFromString(serialized)
        ge_graph = GeGraph(model_def)
        compiled_model = CompiledModel(ge_graph)
        compiled_model._unserialize_graph_attr(model_def)
        return compiled_model

    def compile(self):
        self._initialize()
        self._load()
        logger.info(f'start compile graph: {self._graph.name}.')
        self._executor.compile()
        logger.info(f'end compile graph: {self._graph.name} and start run graph.')

    def _unserialize_graph_attr(self, model_def: ModelDef):
        self._fx_inputs_mapping = unserialize_dict_attr(model_def, "_fx_inputs_mapping")
        self._fx_outputs_mapping = unserialize_dict_attr(model_def, "_fx_outputs_mapping")
        self._local_compile_options = unserialize_dict_attr(model_def, "_local_compile_options")
        self._global_compile_options = unserialize_dict_attr(model_def, "_global_compile_options")
        self._graph_output_ref_input = unserialize_dict_attr(model_def, "_graph_output_ref_input")
        input_process_str = model_def.attr["_inputs_processing"].s.decode("utf-8")
        local_values = {}
        exec(f"input_process = {input_process_str}", globals(), local_values)
        self._input_process = local_values.get("input_process", None)
        if self._input_process is None:
            raise RuntimeError("inputs processing func is not found in model def")

    def _initialize(self):
        initialize_graph_engine(self._global_compile_options)

    def _load(self):
        self._executor.load(self.graph, self._local_compile_options)


def serialize_save_graph(compiled_graph: GeGraph, file_path):
    serialized_compiled_graph = compiled_graph.SerializeToString()
    with open(file_path, "wb") as f:
        f.write(serialized_compiled_graph)


def unserialize_graph(serialized_compiled_graph):
    return CompiledModel.from_serialized_str(serialized_compiled_graph)
