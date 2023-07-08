from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union, Callable
import functools
import threading
import contextlib
import inspect

import torch
from torch.fx.node import Argument, Target
from torch import Tensor

from torchair.core.concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair.core.backend import initialize_graph_engine
from torchair.ge_concrete_graph.ge_ir_pb2 import GraphDef, TensorDescriptor, TensorDef
from torchair.ge_concrete_graph.ge_ir_pb2 import DataType as ProtoDataType
from torchair.ge_concrete_graph.ge_graph import default_ge_graph
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
from torchair.ge_concrete_graph.ge_graph import DataType, TensorSpec
from torchair.ge_concrete_graph.ge_graph import torch_type_to_ge_type, torch_type_to_ge_proto_type
from torchair.core.backend import TorchNpuGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph.utils import convert_to_tensorboard
from torchair.ge_concrete_graph.testing_utils import *
from . import ge_apis as ge

__CONVERTERS = defaultdict(None)
__CONVERTER_TESTS = defaultdict(None)


def _get_converter(name: Callable):
    global __CONVERTERS
    return __CONVERTERS[name]


def _wrap_converter(converter: Callable):
    @functools.wraps(converter)
    def wrapped_converter(*args, **kwargs):
        meta_outputs = None
        if 'meta_outputs' in kwargs:
            meta_outputs = kwargs['meta_outputs']
            if isinstance(meta_outputs, (list, tuple)):
                kwargs['meta_outputs'] = [TensorSpec(v) for v in meta_outputs]
            else:
                kwargs['meta_outputs'] = TensorSpec(meta_outputs)

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
                    assert isinstance(meta_output, torch.Tensor)
                    assert isinstance(ge_output, ge.Tensor)
                    ge_outputs.set_meta(meta_output)

        return ge_outputs
    return wrapped_converter


class Converter:
    compile_backend = None
    result_checker = None

    def __init__(self, aten_op) -> None:
        self._aten_op = aten_op
        self._signature = inspect.signature(aten_op)

        def _(*args, aten_op, **kwargs):
            return aten_op(*args, **kwargs)
        self._eager_func = functools.partial(_, aten_op=self._aten_op)
        self._testcases = None

    def __call__(self, converter) -> Any:
        wrapped_converter = _wrap_converter(converter)
        try:
            self._aten_op._ge_converter = wrapped_converter
        except:
            global __CONVERTERS
            __CONVERTERS.update({self._aten_op: wrapped_converter})
        return self

    @property
    def testcases(self):
        return self._testcases

    @testcases.setter
    def testcases(self, testcases):
        self._testcases = testcases

    def test(self):
        if not hasattr(self, '_compiled_func'):
            self._compiled_func = torch.compile(
                self._eager_func, backend=Converter.compile_backend)
        if self.testcases is None:
            print(f"No testcases for {self._aten_op}", flush=True)
            return

        print(
            f"Testing {self._aten_op} with {len(self.testcases)} cases", flush=True)
        for i, testcase in enumerate(self.testcases):
            print(
                f"[RUN] {self._aten_op} testcase {i + 1}/{len(self.testcases)} with inputs: {testcase}", flush=True)
            args = []
            for arg in testcase.args:
                if isinstance(arg, (list, tuple)):
                    args.append([(v.t() if isinstance(v, TestTensor) else v)
                                 for v in arg])
                elif isinstance(arg, TestTensor):
                    args.append(arg.t())
                else:
                    args.append(arg)
            kwargs = testcase.kwargs
            for k, v in kwargs.items():
                if isinstance(v, TestTensor):
                    kwargs[k] = v.t()

            try:
                backend_results = self._compiled_func(*args, **kwargs)
            except Exception as e:
                backend_results = e

            try:
                eager_results = self._eager_func(*args, **kwargs)
            except Exception as e:
                eager_results = e

            if Converter.result_checker is not None:
                testcase_title = f"{self._aten_op} testcase {i + 1}/{len(self.testcases)} with inputs: {testcase}"
                Converter.result_checker(testcase_title, backend_results, eager_results)
            del args, kwargs, backend_results, eager_results


def register_testcase(testcases: List[TestInput]):
    def add_testcase(converter):
        assert isinstance(converter, Converter)
        converter.testcases = testcases
        global __CONVERTER_TESTS
        __CONVERTER_TESTS.update({converter._aten_op: converter})
        return converter

    return add_testcase


def test_converter(aten_ops: list = None):
    global __CONVERTER_TESTS
    if aten_ops is None:
        for testcase in __CONVERTER_TESTS.values():
            testcase.test()
    elif isinstance(aten_ops, (list, tuple)):
        for aten_op in aten_ops:
            if aten_op not in __CONVERTER_TESTS.keys():
                raise RuntimeError(f"Cannot find testcase for {aten_op}")
            __CONVERTER_TESTS[aten_op].test()
    else:
        assert isinstance(aten_ops, Callable)
        if aten_ops not in __CONVERTER_TESTS.keys():
            raise RuntimeError(f"Cannot find testcase for {aten_ops}")
        __CONVERTER_TESTS[aten_ops].test()


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


class Placement:
    UNDEFINED = -1
    HOST = 0
    DEVICE = 1


class GeConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, graph=None, name=None):
        self._graph = GraphDef() if graph is None else graph
        self._fx_outputs = []
        self._fx_outputs_mapping = dict()
        self._outputs = []
        self._inputs = []
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
        if isinstance(meta_outputs, torch.SymInt):
            data = ge.Data(index=len(self._inputs),
                           dtype=ProtoDataType.DT_INT64, shape=[], name=target)
            data.meta = meta_outputs
            self._inputs.append(data)
            self._input_placements.append(Placement.HOST)
        else:
            assert isinstance(meta_outputs, torch.Tensor)
            dtype = torch_type_to_ge_proto_type(meta_outputs.dtype)
            shape = _get_generalized_shape(meta_outputs)
            data = ge.Data(index=len(self._inputs), dtype=dtype,
                           shape=shape, name=target)
            data.meta = meta_outputs
            self._inputs.append(data)
            self._input_placements.append(Placement.HOST if (
                meta_outputs.device is None or meta_outputs.device.type == 'cpu') else Placement.DEVICE)
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
        return converter(*args, **kwargs, meta_outputs=meta_outputs)

    def dump(self, path: str):
        if path is None:
            return

        if path.endswith(".txt"):
            with open(path, "w+") as f:
                f.write(str(self.graph))
        else:
            try:
                with open(path, "w+") as f:
                    f.write(str(convert_to_tensorboard(self.graph)))
            except Exception as e:
                print(f"dump pbtxt failed {e}", flush=True)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def graph(self):
        return self._graph

    def compile(self) -> Any:
        compile_options = self._config.as_dict()
        logger.info("compile options:")
        for k, v in compile_options.items():
            logger.info(f"  {k}: {v}")

        self.graph.attr["_input_placements"].list.i.extend(
            self._input_placements)
        self.graph.attr["_output_dtypes"].list.i.extend(
            [output.dtype for output in self.outputs])
        _normalize_ge_graph(self.graph)

        initialize_graph_engine()

        self._executor.load(self.graph.SerializeToString(),
                            compile_options)
        self._executor.compile()

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
