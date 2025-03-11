import functools
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import logging
import sys

import torch
from torch._subclasses.fake_tensor import is_fake
import torch.utils._pytree as pytree
from torch.fx import Interpreter
from torch.fx.node import Argument, Target

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable

from torch.profiler import record_function
from torch._dynamo.utils import detect_fake_mode
from torchair.core.utils import logger

aten = torch.ops.aten


class AclGraphRunner():
    def __init__(self, pool=None, stream=None, capture_error_mode: str = "global", num_warmup_iters=0):
        try:
            import torch_npu
        except ImportError:
            raise RuntimeError(
                "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.")

        self._npugraph = torch_npu.npu.NPUGraph()
        self._mempool = torch_npu.npu.graph_pool_handle() if pool is None else pool
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._num_warmup_iters = num_warmup_iters

        self._captured = False
        self._ori_module = None
        self._optimized_module = None

        self._capture_inputs = []
        self._capture_outputs = []
        self._user_inputs_list = []
        self._meta_inputs = []
        self._meta_outputs = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.capture(*args, **kwargs)

        # input process
        for idx in self._user_inputs_list:
            if self._capture_inputs[idx].data_ptr() != args[idx].data_ptr():
                self._capture_inputs[idx].copy_(args[idx])

        # run
        with record_function("acl_graph_replay"):
            self._npugraph.replay()

        return self._capture_outputs

    @property
    def graph(self):
        return self._npugraph

    @property
    def pool(self):
        return self._mempool

    @property
    def stream(self):
        return self._stream

    @property
    def capture_error_mode(self):
        return self._capture_error_mode

    @property
    def num_warmup_iters(self):
        return self._num_warmup_iters

    def capture(self, *args: Any, **kwargs: Any):
        if self._captured:
            # Just capture once now.
            return

        # warm up before capture
        import torch_npu
        torch_npu.npu.synchronize()
        for _ in range(self.num_warmup_iters):
            outs = self._optimized_module(*args, **kwargs)
        torch_npu.npu.synchronize()

        # start capture aclgraph
        self._captured = True
        self._capture_inputs.extend(args)

        with record_function("acl_graph_capture"):
            with torch_npu.npu.graph(self.graph, pool=self.pool, stream=self.stream,
                                     capture_error_mode=self.capture_error_mode):
                self._capture_outputs = self._optimized_module(*args)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('In aclgraph capture, user input list is %s: [%s]', self._user_inputs_list)

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if is_sym(meta_outputs):
            raise AssertionError

        self._meta_inputs.append(args)

        if isinstance(meta_outputs, torch.Tensor):
            if not isinstance(meta_outputs, torch.nn.Parameter):
                self._user_inputs_list.append(len(self._meta_inputs) - 1)
        return meta_outputs

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        # do some optimization in fx for some ops

        return target(*args, **kwargs)

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if is_sym(meta_outputs):
            raise AssertionError

        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise AssertionError

        args = args[0]
        for arg in args:
            self._meta_outputs.append(arg)
        return meta_outputs


class AclGraphInterpreter(Interpreter):
    """
    Interpreter for collect node meta info for acl graph.
    """

    def __init__(self, *args, graph, **kwargs):
        super().__init__(*args, **kwargs)
        self._graph = graph

    def run(self, *args, **kwargs):
        # check args for no sym
        if have_sym_in_gm(self.module):
            raise RuntimeError("Unsupported case in aclgraph: with sym in graph module.")

        # save origin graph module
        self._graph._ori_module = self.module

        # run graph in fake mode
        meta_outs = super().run(*args, **kwargs)

        # do some optimization pass
        optimized_graph_module = self.optimize_graph_module(self.module)
        self._graph._optimized_module = optimized_graph_module

        return self._graph

    def optimize_graph_module(self, graph_module: torch.fx.GraphModule):
        logger.debug('before graph optimization, graph is %s', graph_module.graph)

        # graph optimization passes here, e.g. IFA
        optimized_graph_module = graph_module

        logger.debug('after graph optimization, graph is %s', optimized_graph_module.graph)
        return optimized_graph_module

    def _wrap(self, fn):
        def inner(target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
            func = getattr(super(AclGraphInterpreter, self), fn)
            fake_mode = detect_fake_mode(None)
            with fake_mode:
                meta_outputs = func(target, args, kwargs)
                self._graph.parse_node(target, args, kwargs, meta_outputs)
            return meta_outputs

        return inner

    def run_node(self, n):
        return super().run_node(n)

    def placeholder(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_input = super().placeholder(target, args=args, kwargs=kwargs)
        self._graph.parse_input(target, args, kwargs, meta_input)

        return meta_input

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_function')(target, args, kwargs)

    def call_method(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_method')(target, args, kwargs)

    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        return self._wrap('call_module')(target, args, kwargs)

    def output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        meta_output = super().placeholder(target, args=args, kwargs=kwargs)
        self._graph.parse_output(target, args, kwargs, meta_output)

        return meta_output


def is_sym(v):
    return isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool))


def have_sym_in_gm(graph_module: torch.fx.GraphModule):
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        if is_sym(node.meta['val']):
            return True
    return False
