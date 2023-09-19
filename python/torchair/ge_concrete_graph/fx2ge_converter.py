from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib
import inspect
import sys
import os

import torch
from torch.fx.node import Argument, Target
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor, is_fake
import torch.utils._pytree as pytree

from torchair.core import _torchair
from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair.core.backend import initialize_graph_engine
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import default_ge_graph, GeGraph
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
from torchair.ge_concrete_graph.ge_graph import DataType, TensorSpec
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type
from torchair.ge_concrete_graph.ge_graph import is_sym, sym_to_ge_dtype
from torchair.core.backend import TorchNpuGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph.utils import convert_to_tensorboard
from torchair.ge_concrete_graph.supported_declaration import Support
from . import ge_apis as ge

_CONVERTERS = defaultdict(lambda: None)
_DECLARED_SUPPORTED_CONVERTERS = defaultdict(lambda: None)


def _get_converter(name: Callable):
    if not name in _CONVERTERS:
        from torchair.ge_concrete_graph.ge_converter import custom
        if hasattr(name, "_ge_converter"):
            return name._ge_converter
    return _CONVERTERS[name]


def _wrap_converter(converter: Callable):
    @functools.wraps(converter)
    def wrapped_converter(*args, **kwargs):
        meta_outputs = None
        if 'meta_outputs' in kwargs:
            meta_outputs = kwargs['meta_outputs']
            if isinstance(meta_outputs, (list, tuple)):
                kwargs['meta_outputs'] = [
                    (TensorSpec(v) if v is not None else None) for v in meta_outputs]
            else:
                kwargs['meta_outputs'] = TensorSpec(
                    meta_outputs) if meta_outputs is not None else None

        ge_outputs = converter(*args, **kwargs)

        if meta_outputs is not None:
            if isinstance(ge_outputs, ge.Tensor):
                ge_outputs.set_meta(meta_outputs)
            elif isinstance(ge_outputs, int):
                assert isinstance(meta_outputs, (torch.SymInt, int))
            else:
                assert isinstance(ge_outputs, (list, tuple))
                assert isinstance(meta_outputs, (list, tuple))
                assert len(ge_outputs) == len(meta_outputs)
                for meta_output, ge_output in zip(meta_outputs, ge_outputs):
                    if meta_output is None:
                        continue
                    assert isinstance(meta_output, torch.Tensor)
                    assert isinstance(ge_output, ge.Tensor)
                    ge_output.set_meta(meta_output)

        return ge_outputs
    return wrapped_converter


class ExportSuccess(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return f'ExportSucces: {self.message}'


class Converter:
    compile_backend = None
    result_checker = None

    def __init__(self, aten_op) -> None:
        self._aten_op = aten_op
        self._signature = inspect.signature(aten_op)
        self._supported_cases = None

    def __call__(self, converter) -> Any:
        wrapped_converter = _wrap_converter(converter)
        if 'meta_outputs' in inspect.signature(converter).parameters:
            wrapped_converter.require_meta = True
        else:
            wrapped_converter.require_meta = False
        try:
            self._aten_op._ge_converter = wrapped_converter
        except:
            global _CONVERTERS
            _CONVERTERS.update({self._aten_op: wrapped_converter})
        return self


    @property
    def supported_cases(self):
        return self._supported_cases

    @supported_cases.setter
    def supported_cases(self, supported_cases):
        for testcase in supported_cases:
            assert isinstance(testcase, Support)
        self._supported_cases = supported_cases


def declare_supported(supported_cases: List[Support]):
    def add_testcase(converter):
        assert isinstance(converter, Converter)
        converter.supported_cases = supported_cases
        _DECLARED_SUPPORTED_CONVERTERS.update({converter._aten_op: converter})
        return converter

    return add_testcase


def _declare_supported_converters():
    return _DECLARED_SUPPORTED_CONVERTERS


def register_fx_node_ge_converter(aten_op):
    return Converter(aten_op)


def _get_generalized_shape(fake: torch.Tensor) -> List[int]:
    generalized_shape = []
    for dim in fake.size():
        if not isinstance(dim, torch.SymInt):
            generalized_shape.append(dim)
        else:
            try:
                generalized_shape.append(int(str(dim)))
            except:
                generalized_shape.append(-1)
    return generalized_shape


def _normalize_ge_graph(graph: GraphDef):
    for op in graph.op:
        if op.type == "Data":
            continue
        for desc in op.input_desc:
            if not '_is_unfed_optional' in desc.attr:
                desc.layout = "ND"
                if desc.dtype == ProtoDataType.DT_UNDEFINED:
                    desc.dtype = ProtoDataType.DT_FLOAT
        for desc in op.output_desc:
            desc.layout = "ND"
            if desc.dtype == ProtoDataType.DT_UNDEFINED:
                desc.dtype = ProtoDataType.DT_FLOAT
        if 'input_layout_info' in op.attr:
            indices = op.attr['input_layout_info'].list.i
            layouts = op.attr['input_layout_info'].list.s
            for index, layout in zip(indices, layouts):
                desc = op.input_desc[index]
                if not '_is_unfed_optional' in desc.attr:
                    desc.layout = layout.decode()
        if 'output_layout_info' in op.attr:
            indices = op.attr['output_layout_info'].list.i
            layouts = op.attr['output_layout_info'].list.s
            for index, layout in zip(indices, layouts):
                desc = op.output_desc[index]
                desc.layout = layout.decode()


class Placement:
    UNDEFINED = -1
    HOST = 0
    DEVICE = 1


class ExecutorType:
    CPU = 0
    NPU = 1


def _get_executor_type():
    if 'torch_npu' in sys.modules:
        return ExecutorType.NPU
    return ExecutorType.CPU


def _trans_export_protobuf(inputs, export_graph, file_path, config):
    weight_name = config.export_config.weight_name
    inputs_name = config.export_config.inputs_name
    num_weight_in_graph = 0
    for i, inp in enumerate(inputs):
        if id(inp) in weight_name:
            logger.debug(f'  Weight {i} dtype: {inp.dtype} shape: {inp.shape}')
            file_id = weight_name[id(inp)]
            y = ge.FileConstant(shape=list(inp.shape),
                                dtype=torch_type_to_ge_type(inp.dtype),
                                file_path=file_path + "/" + file_id.replace(".", "_"),
                                node_name=export_graph.op[i].name)
            export_graph.op[i].Clear()
            export_graph.op[i].MergeFrom(y._node)
            num_weight_in_graph += 1
        if id(inp) in inputs_name:
            logger.debug(f'  Input {i} dtype: {inp.dtype} shape: {inp.shape}')
            export_graph.op[i].attr["use_define_name"].s = compat_as_bytes(inputs_name[id(inp)])
    return num_weight_in_graph


def _save_weight2file(inputs, file_path, weight_name, num_weight_in_graph):
    logger.info(f'save Weight tensor to file...')
    saved_num = 0
    for i, inp in enumerate(inputs):
        if id(inp) in weight_name:
            file_id = weight_name[id(inp)]
            if inp.is_cpu:
                inp.numpy().tofile(file_path + "/" + file_id.replace(".", "_"))
            else:
                inp.cpu().numpy().tofile(file_path + "/" + file_id.replace(".", "_"))
            saved_num += 1
            print('\r torchair dynamo export save weight {0}% {1}/{2}'.format(
                  min(100, int(saved_num / num_weight_in_graph * 100)), saved_num, num_weight_in_graph), end='')
    print(" ")
    logger.info(f'save Weight tensor to file over...')


def dump(path: str, graph):
    if path is None:
        return

    if path.endswith(".txt"):
        with open(path, "w+") as f:
            f.write(str(graph))
    elif path.endswith('.py'):
        with open(path, "w+") as f:
            f.write(str(graph.python_code))
    else:
        try:
            with open(path, "w+") as f:
                f.write(str(convert_to_tensorboard(graph)))
        except Exception as e:
            print(f"dump pbtxt failed {e}", flush=True)


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, graph=None, name=None):
        self._graph = GeGraph() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []
        self._fx_inputs_mapping = dict()
        self._input_placements = []
        self._executor = TorchNpuGraph(name)
        self._config = config
        self._auto_tune_times = 0

    def context(self):
        return default_ge_graph(self.graph)

    @property
    def is_dynamic(self):
        if hasattr(self, '_is_dynamic'):
            return self._is_dynamic

        for input in self._inputs:
            if (-1 in input.desc.shape.dim) or (-2 in input.desc.shape.dim):
                self._is_dynamic = True
                return True

        self._is_dynamic = False
        return False

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if is_sym(meta_outputs):
            data = ge.Data(index=len(self._inputs),
                           dtype=sym_to_ge_dtype(meta_outputs), shape=[], placement='CPU', name=target)
            data.set_meta(meta_outputs)
            self._inputs.append(data)
            self._input_placements.append(Placement.HOST)
        else:
            assert isinstance(meta_outputs, torch.Tensor)
            dtype = torch_type_to_ge_type(meta_outputs.dtype)
            shape = _get_generalized_shape(meta_outputs)
            placement = Placement.HOST if (
                meta_outputs.device is None or meta_outputs.device.type == 'cpu') else Placement.DEVICE
            data = ge.Data(index=len(self._inputs), dtype=dtype,
                           shape=shape, placement='CPU' if (placement == Placement.HOST) else 'NPU', name=target)
            data.set_meta(meta_outputs)
            self._inputs.append(data)
            self._input_placements.append(placement)
        return self._inputs[-1]

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        assert isinstance(args, (list, tuple)) and len(args) == 1
        args = args[0]
        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            if isinstance(arg, ge.Tensor):
                self._fx_outputs_mapping[len(
                    self._fx_outputs)] = len(self._outputs)
                self._outputs.append(arg)
            self._fx_outputs.append(arg)

        ge.NetOutput(self._outputs)
        return args

    def parse_symlist(self, syms):
        npu_syms = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                npu_syms.append(sym.npu)
            else:
                assert isinstance(sym, int)
                npu_syms.append(sym)
        if all([isinstance(sym, int) for sym in npu_syms]):
            return npu_syms
        return ge.Pack(npu_syms, N=len(npu_syms), axis=0)

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if hasattr(target, "_ge_converter"):
            converter = target._ge_converter
        else:
            converter = _get_converter(target)
        if converter is None:
            raise RuntimeError(f"Unsupported torch op {target} by ge")
        if converter.require_meta:
            return converter(*args, **kwargs, meta_outputs=meta_outputs)
        else:
            return converter(*args, **kwargs)

    def dump(self, path: str):
        dump(path, self.graph)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    def set_fx_inputs_mapping(self, fx_inputs_mapping):
        self._fx_inputs_mapping = fx_inputs_mapping

    def compile(self) -> Any:
        compile_options = self._config.as_dict()
        compile_options["ge.exec.formatMode"] = "1"
        logger.info("compile options:")
        for k, v in compile_options.items():
            logger.info(f"  {k}: {v}")

        self.graph.attr["_input_placements"].list.i.extend(
            self._input_placements)
        self.graph.attr["_output_dtypes"].list.i.extend(
            [output.dtype for output in self.outputs])
        self.graph.attr["_executor_type"].i = _get_executor_type()

        self._complement_graph_attr()

        _normalize_ge_graph(self.graph)

        initialize_graph_engine()

        self._executor.load(self.graph.SerializeToString(),
                            compile_options)
        self._executor.compile()

    def export(self, inputs) -> Any:
        compile_options = self._config.as_dict()
        logger.info("export options:")
        for k, v in compile_options.items():
            logger.info(f"  {k}: {v}")

        file_path = self._config.export_config.export_path_dir
        export_graph = GeGraph()
        export_graph.MergeFrom(self._graph._proto)

        if os.path.exists(file_path) is False:
            os.mkdir(file_path)

        num_weight_in_graph = _trans_export_protobuf(inputs, export_graph, file_path, self._config)

        _save_weight2file(inputs, file_path, self._config.export_config.weight_name, num_weight_in_graph)
        
        _normalize_ge_graph(export_graph)

        dump(self._config.debug.graph_dump.full_path(file_path + "/dynamo"), export_graph)

        _torchair.export(export_graph.SerializeToString(), compile_options)

    @property
    def should_auto_tune(self) -> bool:
        if self._config.aoe_config.aoe_mode.value is None:
            return False
        if self.is_dynamic:
            # AOE is not supported for dynamic shape now
            return False
        return self._auto_tune_times == 0

    def auto_tune(self, inputs: List[Tensor]) -> Any:
        logger.info(f"Start auto tune for round {self._auto_tune_times}")
        self._executor.auto_tune(inputs)
        self._auto_tune_times += 1
        logger.info(f"End auto tune for round {self._auto_tune_times - 1}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = [(arg if arg.is_contiguous() else arg.contiguous()) if isinstance(arg, torch.Tensor)
                  else torch.tensor(arg) for arg in args]

        ge_inputs = [None for v in self._fx_inputs_mapping]
        for fx_idx, ge_idx in self._fx_inputs_mapping.items():
            ge_inputs[ge_idx] = inputs[fx_idx]
        inputs = ge_inputs

        self._consume_data_into_inputs(inputs)

        if self._config.export_config.export_mode:
            self.export(inputs)
            raise ExportSuccess("export graph over")

        if self.should_auto_tune:
            self.auto_tune(inputs)

        ge_outputs = self._executor.run(inputs)
        assert len(ge_outputs) == len(
            self._fx_outputs_mapping), f"output size mismatch, expect {len(self._fx_outputs_mapping)}, got {len(ge_outputs)}"

        fx_outputs = [v for v in self._fx_outputs]
        for fx_idx, ge_idx in self._fx_outputs_mapping.items():
            assert ge_idx < len(
                ge_outputs), f"output index {ge_idx} out of range {len(ge_outputs)}"
            fx_outputs[fx_idx] = ge_outputs[ge_idx]

        del ge_outputs

        return tuple(fx_outputs)

    def _complement_graph_attr(self):
        num_inputs = self.graph.num_inputs()
        diff = num_inputs - len(self._inputs)
        self._inputs.extend([None for _ in range(diff)])
        self._input_placements.extend([None for _ in range(diff)])
        self.graph.attr["_input_placements"].list.i.extend(
            [-1 for _ in range(diff)])

        for gen in self.graph.generator_rng_state:
            rng_state = self.graph.get_graph_rng_state(gen)
            idx, offset_data = rng_state.get_idx_and_offset()
            assert len(self._inputs) == len(self._input_placements)
            placement = Placement.HOST
            self._inputs[idx] = offset_data
            self._input_placements[idx] = placement
            self.graph.attr["_input_placements"].list.i[idx] = placement

    def _consume_data_into_inputs(self, inputs):
        num_inputs = self.graph.num_inputs()
        diff = num_inputs - len(inputs)
        inputs.extend([None for _ in range(diff)])
        for gen in self.graph.generator_rng_state:
            rng_state = self.graph.get_graph_rng_state(gen)
            idx, offset = rng_state.consume()
            inputs[idx] = offset