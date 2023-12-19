from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
from datetime import datetime
import functools
import threading
import contextlib
from contextlib import contextmanager
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
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef, OpDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import default_ge_graph, GeGraph
from torchair.ge_concrete_graph.ge_graph import Tensor as GeTensor
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
from torchair.ge_concrete_graph.ge_graph import DataType, TensorSpec
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type
from torchair.ge_concrete_graph.ge_graph import is_sym, sym_to_ge_dtype
from torchair.core.backend import TorchNpuGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph.utils import convert_to_tensorboard, force_op_unknown_shape, dump_graph
from torchair.ge_concrete_graph.supported_declaration import Support
from torchair.ge_concrete_graph.export_config_generete import generate_config
from torchair.utils.export_utils import make_export_graph, get_export_file_name
from . import ge_apis as ge


def _mapping_assign_op_to_graph_output(graph: GraphDef):
    net_output: OpDef = None  # 输出节点
    net_inputs: Dict[str, int] = {}  # 输入tensor名称到索引的映射

    for op in graph.op:
        if op.type == "Data":
            net_inputs[GeTensor(op).tensor] = op.attr["index"].i
        elif op.type == "NetOutput":
            net_output = op
    assert net_output is not None, "NetOutput not found"

    def _mapping_to_graph_output(graph: GraphDef, graph_out: OpDef, assign_node_out: GeTensor, value_tensor: str):
        for i, name in enumerate(graph_out.input):
            if name == assign_node_out.tensor:
                graph_out.input[i] = value_tensor
                return i
        graph_out.input.append(value_tensor)
        graph_out.input_desc.add().CopyFrom(assign_node_out.desc)
        graph_out.input_desc[-1].name = f"input{len(graph_out.input_desc) - 1}"
        graph.attr["_output_dtypes"].list.i.append(assign_node_out.dtype)
        return len(graph_out.input) - 1

    output_refto_input = {}
    replaced_assign_ops = []
    for op in graph.op:
        if op.type != "Assign":
            continue
        assign_node_out = GeTensor(op)
        logger.info(f"Found assign op {assign_node_out}")
        if op.input[0] in net_inputs.keys(): # Assign在给输入赋值
            logger.info(
                f"Replace assign op {op.name} assign value from {op.input[1]} to input {net_inputs[op.input[0]]} {op.input[0]}")
            output_index = _mapping_to_graph_output(graph, net_output, assign_node_out, op.input[1])
            output_refto_input[output_index] = net_inputs[op.input[0]]
            replaced_assign_ops.append(op)
        else:
            logger.info(f"Collect assign op {op.name} assign value from {op.input[1]} to {op.input[0]}")
            net_output.input.append(assign_node_out.controller)

    for op in replaced_assign_ops:
        graph.op.remove(op)

    return output_refto_input

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


def _update_internal_format_from_inputs(graph: GraphDef, runtime_inputs):
    if 'torch_npu' not in sys.modules:
        logger.info(f'The internal format will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip internal format updates')
        return

    torch_npu_module = sys.modules['torch_npu']
    input_index_mapping_graph_op: Dict[int, OpDef] = {}  # data节点的index到opDef的映射
    for op in graph.op:
        if op.type == "Data":
            input_index_mapping_graph_op[op.attr["index"].i] = op

    for idx in range(len(runtime_inputs)):
        assert idx < len(input_index_mapping_graph_op), \
            f"GE graph input index {idx} out of Data ops index range {len(input_index_mapping_graph_op)}"

        # attr "format_for_int" in proto::TensorDescriptor will be be deserialized as TensorDesc Format in ge.
        input_index_mapping_graph_op[idx].output_desc[0].attr["format_for_int"].i = \
            torch_npu_module.get_npu_format(runtime_inputs[idx])
        logger.info(f'update the Format of output TensorDesc for input_{idx} '
                    f'to Format {torch_npu_module.get_npu_format(runtime_inputs[idx])}')


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


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, graph=None, name=None):
        self._graph = GeGraph() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []
        self._fx_inputs_mapping = dict()
        self._input_placements = []
        self._graph_output_ref_input = {}
        self._executor = TorchNpuGraph(name)
        self._config = config
        self._auto_tune_times = 0
        self._converter_ctx = threading.local()
        self._fx_input_mapping_cloned_ge_input = []
        self._inputs_processing = None
        self._is_compiled = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._inputs_processing is None:
            self._inputs_processing = self._make_inputs_processing_func(*args)
        inputs = self._inputs_processing(*args)

        inputs = self._consume_data_into_inputs(inputs)

        if self.config.export_config.export_mode:
            self.export(inputs)
            raise ExportSuccess("export graph over")

        self.load(inputs)

        if self.should_auto_tune:
            self.auto_tune(inputs)

        self.compile()

        if len(self._graph_output_ref_input):
            assigned_outputs = [None] * len(self.graph.attr["_output_dtypes"].list.i)
            for output_index, input_index in self._graph_output_ref_input.items():
                assigned_outputs[output_index] = inputs[input_index]
            ge_outputs = self._executor.run(inputs, assigned_outputs)

            for index_tuple in self._fx_input_mapping_cloned_ge_input:
                args[index_tuple[0]].copy_(inputs[index_tuple[1]])
        else:
            ge_outputs = self._executor.run(inputs)

        assert len(ge_outputs) == len(self.graph.attr["_output_dtypes"].list.i),\
            f"output size mismatch, expect {len(self.graph.attr['_output_dtypes'].list.i)}, got {len(ge_outputs)}"

        fx_outputs = [v for v in self._fx_outputs]
        for fx_idx, ge_idx in self._fx_outputs_mapping.items():
            assert ge_idx < len(ge_outputs), f"output index {ge_idx} out of range {len(ge_outputs)}"
            fx_outputs[fx_idx] = ge_outputs[ge_idx]

        del ge_outputs

        return tuple(fx_outputs)

    def load(self, runtime_inputs) -> Any:
        if self._is_compiled:
            return

        # Initialize based on global options
        local_compile_options, global_compile_options = self.config.as_dict()
        global_compile_options["ge.exec.staticMemoryPolicy"] = "2"
        logger.info("global compile options:")
        for k, v in global_compile_options.items():
            logger.info(f"  {k}: {v}")

        initialize_graph_engine(global_compile_options)

        # Update local options
        self._graph_output_ref_input = _mapping_assign_op_to_graph_output(self.graph)
        if self.config.debug.graph_dump.enabled and len(self._graph_output_ref_input):
            self.dump(f'dynamo_after_mapping_assign_{datetime.now().strftime("%Y%m%d%H%M%S%f")}.pbtxt')
        output_reuse_indexes = [x for x in range(len(self.outputs)) if x not in self._graph_output_ref_input.keys()]
        if len(output_reuse_indexes) != 0:
            # support output memory reuse while output is not ref to input
            local_compile_options["ge.exec.outputReuseMemIndexes"] = ",".join(str(x) for x in output_reuse_indexes)
        logger.info("local compile options:")
        for k, v in local_compile_options.items():
            logger.info(f"  {k}: {v}")

        # Normalize graph
        self.graph.attr["_input_placements"].list.i.extend(self._input_placements)
        self.graph.attr["_output_dtypes"].list.i.extend([output.dtype for output in self.outputs])
        self.graph.attr["_executor_type"].i = _get_executor_type()
        self._complement_graph_attr()

        _normalize_ge_graph(self.graph)

        _update_internal_format_from_inputs(self.graph, runtime_inputs)

        self._executor.load(self.graph.SerializeToString(), local_compile_options)

    def compile(self) -> Any:
        if self._is_compiled:
            return

        logger.info(f'start compile graph: {self}.')
        self._executor.compile()
        self._is_compiled = True
        logger.info(f'end compile graph: {self} and start run graph.')

    def context(self):
        return default_ge_graph(self.graph)

    @contextmanager
    def converter_context(self, *, node):
        try:
            self._converter_ctx.node = node
            yield
        finally:
            self._converter_ctx.node = None

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
                           dtype=sym_to_ge_dtype(meta_outputs), shape=[], placement='CPU', node_name=target)
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
                           shape=shape, placement='CPU' if (placement == Placement.HOST) else 'NPU', node_name=target)
            data.set_meta(meta_outputs)
            self._inputs.append(data)
            self._input_placements.append(placement)
        return self._inputs[-1]

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        assert isinstance(args, (list, tuple)) and len(args) == 1
        args = args[0]
        for arg in args:
            arg = arg.npu if isinstance(arg, ValuePack) else arg
            self._fx_outputs.append(arg)
            if not isinstance(arg, ge.Tensor):
                continue

            output_index = len(self._outputs)
            for i, output in enumerate(self._outputs):
                if output.tensor == arg.tensor:
                    output_index = i
                    break
            self._fx_outputs_mapping[len(
                self._fx_outputs) - 1] = output_index
            if output_index == len(self._outputs):
                self._outputs.append(arg)

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
        # force unknown shape with ge.Pack when parse symlist
        return force_op_unknown_shape(ge.Pack(npu_syms, N=len(npu_syms), axis=0))

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if hasattr(target, "_ge_converter"):
            converter = target._ge_converter
        else:
            converter = _get_converter(target)
        if converter is None:
            raise RuntimeError(f"Unsupported torch op {target} by ge")
        if converter.require_meta:
            ge_outputs = converter(*args, **kwargs, meta_outputs=meta_outputs)
            if meta_outputs is not None and hasattr(self._converter_ctx, 'node') and self._converter_ctx.node:
                fx_tensor_prefix = f'{self._converter_ctx.node.name}-{self._converter_ctx.node.target}.OUTPUT'
                if isinstance(ge_outputs, ge.Tensor):
                    ge_outputs.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.0')
                elif isinstance(ge_outputs, (list, tuple)) and all([isinstance(v, ge.Tensor) for v in ge_outputs]):
                    for i, ge_output in enumerate(ge_outputs):
                        ge_output.desc.attr["_fx_tensor_name"].s = compat_as_bytes(f'{fx_tensor_prefix}.{i}')

            return ge_outputs
        else:
            return converter(*args, **kwargs)

    def dump(self, path: str):
        dump_graph(path, self.graph)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    @property
    def config(self):
        return self._config

    def set_fx_inputs_mapping(self, fx_inputs_mapping):
        self._fx_inputs_mapping = fx_inputs_mapping

    def export(self, inputs) -> Any:
        file_path = self.config.export_config.export_path_dir
        file_name_air = get_export_file_name(self.config.export_config.export_name)

        export_graph = make_export_graph(inputs, self.config, self.graph)

        _normalize_ge_graph(export_graph)

        if self.config.export_config.auto_atc_config_generated:
            generate_config(self.config, file_path, export_graph)
        local_options = {}
        local_options["export_path_dir"] = file_path
        local_options["export_name"] = file_name_air
        _torchair.export(export_graph.SerializeToString(), local_options)

    @property
    def should_auto_tune(self) -> bool:
        if self.config.aoe_config.aoe_mode.value is None:
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
        if diff > 0:
            istuple = isinstance(inputs, tuple)
            inputs = list(inputs)
            inputs.extend([None for _ in range(diff)])
            for gen in self.graph.generator_rng_state:
                rng_state = self.graph.get_graph_rng_state(gen)
                idx, offset = rng_state.consume()
                inputs[idx] = offset
            inputs = tuple(inputs) if istuple else inputs
        return inputs

    def _make_inputs_processing_func(self, *args: Any):
        uncontiguous_ge_input_idx = []
        nontensor_ge_input_idx = []
        for fx_index, ge_index in self._fx_inputs_mapping.items():
            if isinstance(args[fx_index], torch.Tensor):
                if not args[fx_index].is_contiguous():
                    uncontiguous_ge_input_idx.append(ge_index)
                    self._fx_input_mapping_cloned_ge_input.append((fx_index, ge_index))
            else:
                nontensor_ge_input_idx.append(ge_index)

        if len(self._fx_inputs_mapping) == len(args):
            if len(uncontiguous_ge_input_idx) == 0 and len(nontensor_ge_input_idx) == 0:
                def inputs_processing(*args: Any):
                    return args
            else:
                def inputs_processing(*args: Any):
                    inputs = list(args)
                    for idx in uncontiguous_ge_input_idx:
                        inputs[idx] = inputs[idx].contiguous()
                    for idx in nontensor_ge_input_idx:
                        inputs[idx] = torch.tensor(inputs[idx])
                    return inputs
        else:
            def inputs_processing(*args: Any):
                inputs = [None] * len(self._fx_inputs_mapping)
                for fx_idx, ge_idx in self._fx_inputs_mapping.items():
                    inputs[ge_idx] = args[fx_idx]

                for idx in uncontiguous_ge_input_idx:
                    inputs[idx] = inputs[idx].contiguous()
                for idx in nontensor_ge_input_idx:
                    inputs[idx] = torch.tensor(inputs[idx])
                return inputs

        return inputs_processing
