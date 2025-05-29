from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import functools
import logging
import sys

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.node import Argument, Target
from torch.profiler import record_function

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable

from torchair.configs.compiler_config import CompilerConfig
from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair._acl_concrete_graph.acl_graph import have_sym_in_list
from torchair._utils.path_manager import PathManager

aten = torch.ops.aten


class AclConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, name="graph", pool=None, stream=None, capture_error_mode: str = "global",
                 num_warmup_iters=0):
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e

        self._config = config
        self._graph_name = name
        self._npugraph = torch_npu.npu.NPUGraph()
        self._mempool = torch_npu.npu.graph_pool_handle() if pool is None else pool
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._num_warmup_iters = num_warmup_iters

        self._captured = False
        self._fx_graph = None
        self._replay_func: Callable = None

        self._capture_inputs = []
        self._capture_outputs = []
        self._user_inputs_mapping = OrderedDict()
        self._meta_inputs = []
        self._meta_outputs = []

        self._mutated_user_inputs = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.compile(*args, **kwargs)

        # input process
        for idx in self._user_inputs_mapping.values():
            if self._capture_inputs[idx].data_ptr() != args[idx].data_ptr():
                self._capture_inputs[idx].copy_(args[idx])

        # run
        with record_function("acl_graph_replay"):
            self._replay_func(*args, **kwargs)

        # For in-place op, dynamo will transform it into a functionalized call and add copy_ node when setting
        # keep_inference_input_mutations=True, which may need data copy from capture input to user input (when tensor
        # address is different between capture and replay).
        for arg_name in self._mutated_user_inputs:
            if arg_name not in self._user_inputs_mapping:
                raise RuntimeError(f"{arg_name} is not in input args: {self._user_inputs_mapping.keys()}")
            idx = self._user_inputs_mapping[arg_name]
            if self._capture_inputs[idx].data_ptr() != args[idx].data_ptr():
                logger.warning_once(f"Mutated input[{arg_name}]'s data_ptr is different between capture and replay. "
                                    f"This may call redundant copy. ")
                args[idx].copy_(self._capture_inputs[idx])

        return self._capture_outputs

    @property
    def config(self):
        return self._config

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

    @property
    def fx_graph(self):
        return self._fx_graph

    def save_fx_graph(self, graph_module: torch.fx.GraphModule):
        self._fx_graph = graph_module

    @contextmanager
    def context(self):
        # TO DO: add device context manager for acl graph
        try:
            yield
        finally:
            pass

    @contextmanager
    def converter_context(self, *, node):
        try:
            yield
        finally:
            pass

    def dump(self, path: str):
        if path is None:
            raise RuntimeError("Path is none, please report a bug.")
        if not path.endswith('.py'):
            raise NotImplementedError(
                f"Graph dump for aclGraph only support 'py' type, but got: {self.config.debug.graph_dump.type.value}."
                f"Please check compile config setting: config.debug.graph_dump.type")
        else:
            PathManager.check_path_writeable_and_safety(path)
            with open(path, "w+") as f:
                f.write(self.fx_graph.print_readable(False))


    def codegen(self, extend_config, enable_cache=False):
        raise NotImplementedError("Codegen for acl graph is not implemented!")

    def optimize_graph_without_runtime(self, *sample_args):
        logger.debug('before graph optimization, graph is %s', self.fx_graph.graph)

        # graph optimization passes here
        # Note: this pass need sample args to run in FakeTensor mode, any pass modifies ops without meta registration
        # should run after it.
        if not self.config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass:
            logger.debug("Start to process reinplace inplaceable ops fx pass for graph: %s", id(self.fx_graph))
            from torchair._acl_concrete_graph.graph_pass import _reinplace_inplaceable_ops_pass
            _reinplace_inplaceable_ops_pass(self.fx_graph, *sample_args)

        from torchair._acl_concrete_graph.acl_graph import replace_dynamic_workspace_ops, _find_mutated_user_inputs
        replace_dynamic_workspace_ops(self.fx_graph)

        # Note: this will modify mutated input ops in fx graph, should be executed LAST.
        if not self.config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass:
            logger.debug("Start to process reinplace input mutated ops fx pass for graph: %s", id(self.fx_graph))
            from torchair._acl_concrete_graph.graph_pass import _reinplace_input_mutated_ops
            _reinplace_input_mutated_ops(self.fx_graph)

        # find mutated inputs after graph optimization passes
        self._mutated_user_inputs = _find_mutated_user_inputs(self.fx_graph)
        logger.debug('find mutated user inputs: %s', self._mutated_user_inputs)
        logger.debug('after graph optimization, graph is %s', self.fx_graph.graph)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path(f"dynamo_optimized_{self._graph_name}"))

    def compile(self, *args: Any, **kwargs: Any):
        if self._captured:
            # A fx graph just be captured once now.
            return

        import torch_npu
        # warm up before capture
        with record_function("acl_graph_warm_up"):
            for _ in range(self.num_warmup_iters):
                torch_npu.npu.synchronize()
                outs = self.fx_graph(*args, **kwargs)
                torch_npu.npu.synchronize()

        # start capture aclgraph
        self._captured = True
        self._capture_inputs.extend(args)

        logger.debug('Start to capture fx graph[id: %s] for AclGraph[id: %s].', id(self.fx_graph), id(self.graph))
        with record_function("acl_graph_capture"):
            self.capture(*args, **kwargs)
        logger.info('Success to capture fx graph[id: %s] and start to run AclGraph[id: %s].',
                    id(self.fx_graph), id(self.graph))

    def capture(self, *args: Any, **kwargs: Any):
        from torchair._acl_concrete_graph.acl_graph import (UpdatedNodeCaptureInterp, CapturedGraphUpdateAndReplay,
                                                            CapturedGraph)
        captured_interpreter = UpdatedNodeCaptureInterp(self.fx_graph, self._meta_inputs)

        updated_input_func = captured_interpreter.process_need_updated_ops()

        with CapturedGraph(self.graph, pool=self.pool, stream=self.stream,
                           capture_error_mode=self.capture_error_mode, fx_graph=self.fx_graph):
            self._capture_outputs = captured_interpreter.run(*args, **kwargs)
        updated_node_infos = captured_interpreter.captured_node_infos
        logger.debug('In graph {%s}, the updated node num is {%s}.', id(self.fx_graph), len(updated_node_infos))

        # gen run func
        self._replay_func = CapturedGraphUpdateAndReplay(self.graph, updated_input_func, updated_node_infos)
        logger.debug('In graph {%s}, all the non parameter tensor input index list is: {%s}.',
                     id(self.fx_graph), self._user_inputs_mapping.values())

    def parse_symlist(self, syms):
        npu_syms = []
        for sym in syms:
            if isinstance(sym, ValuePack):
                npu_syms.append(sym.npu)
            else:
                if not isinstance(sym, int):
                    raise RuntimeError(f"Unsupported case with non constant value [{sym}] in sym_list [{syms}].")
                npu_syms.append(sym)
        if all([isinstance(sym, int) for sym in npu_syms]):
            return npu_syms

        logger.debug("Node inputs have symbol[%s] in acl graph.", npu_syms)
        return npu_syms

    def parse_input(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        self._meta_inputs.append(meta_outputs)

        # Lazy check for int/sym inputs
        if isinstance(meta_outputs, torch.Tensor):
            meta_shape = list(meta_outputs.shape)
            if have_sym_in_list(meta_shape):
                # Sym in tensor means dynamic shape, it is unsupported.
                raise RuntimeError(f"Unsupported case in AclGraph: with sym in graph input tensor[{meta_outputs}].")

            if not isinstance(meta_outputs, torch.nn.Parameter):
                self._user_inputs_mapping.setdefault(target, len(self._meta_inputs) - 1)

        return meta_outputs

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        # do some optimization in fx for some ops

        return target(*args, **kwargs)

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise RuntimeError(f"Unsupported case in AclGraph: for output node with args: [{args}].")

        args = args[0]
        for arg in args:
            self._meta_outputs.append(arg.meta)

            meta_shape = list(self._meta_outputs[-1].shape)
            if have_sym_in_list(meta_shape):
                raise RuntimeError(
                    f"Unsupported case in AclGraph: with sym in output tensor: [{self._meta_outputs[-1]}].")

        return meta_outputs
