__all__ = []

import contextlib
import functools
import inspect
from collections import defaultdict
from typing import List, Callable, Any, Dict, Tuple, Union

import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._tensor import Tensor
from torch.utils._mode_utils import no_dispatch
from torch.fx import Interpreter
from torch.fx.node import Argument, Target
from torch._functorch.aot_autograd import aot_module_simplified

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable
    
from torch._dynamo.utils import detect_fake_mode
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._subclasses.fake_tensor import FakeTensor

from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack, _is_symlist
from torchair.core.utils import logger
from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph as ConcreteGraph
from torchair.configs.compiler_config import CompilerConfig
from torchair._ge_concrete_graph.fx2ge_converter import _get_converter
from torchair._utils.path_manager import PathManager


aten = torch.ops.aten


@contextlib.contextmanager
def _pretty_faketensor():

    def fake_tensor_repr(self):
        return f"{str(self.dtype).split('.')[-1]}{tuple(self.size())}"

    prior = FakeTensor.__repr__
    try:
        FakeTensor.__repr__ = fake_tensor_repr
        yield
    finally:
        FakeTensor.__repr__ = prior


def _contain_statement(func, statement):
    source_code = inspect.getsource(func)
    return statement in source_code


def _get_converter_status(converter):
    if converter is None:
        return '未注册'

    somecase_unsupported = False
    somecase_supported = False
    if hasattr(converter, '__wrapped__') and _contain_statement(converter.__wrapped__, 'raise NotImplementedError'):
        somecase_unsupported = True

    if _contain_statement(converter, 'return '):
        somecase_supported = True

    if somecase_unsupported and somecase_supported:
        return "部分支持"
    elif somecase_unsupported:
        return "未实现"
    elif somecase_supported:
        return "已支持"
    else:
        return "未知状态，请查看源码"


class _OpSummary:
    def __init__(self, op) -> None:
        self._op = op
        self._count = 0
        self._inputs = dict()
        self._outputs = dict()

        if is_builtin_callable(op):
            self._type = 'builtin'
            self._status = '已支持'
        else:
            self._type = 'prims' if 'prim' in str(op) else 'aten'
            if hasattr(op, "_ge_converter"):
                converter = op._ge_converter
            else:
                converter = _get_converter(op)
            self._status = _get_converter_status(converter)

    def record(self, args, kwargs, output):
        with _pretty_faketensor():
            output_str = f"{output}"
            if len(kwargs):
                input_str = f"{args}, {kwargs}"
            else:
                input_str = f"{args}"
        self._inputs[input_str] = self._inputs.get(
            input_str, 0) + 1
        self._outputs[output_str] = self._outputs.get(output_str, 0) + 1
        self._count += 1

    def summary(self):
        sorted_inputs = sorted(self._inputs.items(),
                               key=lambda x: x[1], reverse=True)
        sorted_outputs = sorted(self._outputs.items(),
                                key=lambda x: x[1], reverse=True)
        input_str = '\n'.join(
            [f"{item[1]}次：{item[0]}" for item in sorted_inputs])
        output_str = '\n'.join(
            [f"{item[1]}次：{item[0]}" for item in sorted_outputs])

        return [str(self._op), self._type, self._status, str(self._count), input_str, output_str]


class _SummarizeFxGraph(Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._summary = {}

    def summary(self):
        return [v.summary() for v in self._summary.values()]

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        output = super().call_function(target, args, kwargs)
        if target not in self._summary:
            self._summary[target] = _OpSummary(target)
        self._summary[target].record(args, kwargs, output)

        return output


def _summarize_fx_graph(graph, example_inputs, csv_file: str = None):
    if csv_file is None:
        return

    interpreter = _SummarizeFxGraph(graph)
    fake_mode = detect_fake_mode(None)
    with fake_mode:
        interpreter.run(*example_inputs)

    if not csv_file.endswith(".csv"):
        raise AssertionError("file must be a csv file")
    
    try:
        import csv
    except ImportError as e:
        raise ImportError("Please install csv to use csv file to save summary") from e
    PathManager.check_path_writeable_and_safety(csv_file)
    with open(csv_file, 'w+', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["目标函数", "函数类型", "支持状态", "调用次数", "输入统计", "输出统计"])
        writer.writerows(interpreter.summary())
