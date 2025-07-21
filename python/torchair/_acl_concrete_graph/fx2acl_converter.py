from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import functools
import logging
import sys
import weakref

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.node import Argument, Target
from torch.profiler import record_function
from torch.types import Device, Number

try:
    from torch._dynamo.allowed_functions import is_builtin_callable
except ModuleNotFoundError:
    from torch._dynamo.trace_rules import is_builtin_callable

from torchair.configs.compiler_config import CompilerConfig
from torchair.core._concrete_graph import ConcreteGraphBase, ValuePack
from torchair.core.utils import logger
from torchair._acl_concrete_graph.acl_graph import have_sym_in_list
from torchair._utils.path_manager import PathManager
from torchair._acl_concrete_graph.graph_pass import apply_event_closure_with_multi_stream
from torchair._acl_concrete_graph.utils import (debug_mem_state, WeakRef, GraphMeta, get_tensor_metadata,
                                                reconstruct_from_tensor_metadata, reconstruct_args_kwargs)

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
        self._mempool = torch_npu.npu.graph_pool_handle() if pool is None else pool
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._num_warmup_iters = num_warmup_iters
        self._device = torch_npu.npu.current_device()

        self._captured = False
        self._fx_graph = None
        self._meta_inputs = []
        self._meta_outputs = []
        self._mutated_user_inputs = []
        self._user_inputs_mapping = OrderedDict()
        self._unupdated_input_func = None
        self._updated_input_func = None
        self._updated_ops_param = None

        self._original_mem_state = None
        self._graphs_meta: Dict[str, GraphMeta] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn_key = self.compile(*args, **kwargs)

        # input process
        for idx in self._user_inputs_mapping.values():
            if self._graphs_meta[fn_key].captured_inputs[idx].data_ptr() != args[idx].data_ptr():
                self._graphs_meta[fn_key].captured_inputs[idx].copy_(args[idx])

        # run
        with record_function("acl_graph_replay"):
            self._graphs_meta[fn_key].replay_func(*args, **kwargs)

        # For in-place op, dynamo will transform it into a functionalized call and add copy_ node when setting
        # keep_inference_input_mutations=True, which may need data copy from capture input to user input (when tensor
        # address is different between capture and replay).
        for arg_name in self._mutated_user_inputs:
            if arg_name not in self._user_inputs_mapping:
                raise RuntimeError(f"{arg_name} is not in input args: {self._user_inputs_mapping.keys()}")
            idx = self._user_inputs_mapping[arg_name]
            if self._graphs_meta[fn_key].captured_inputs[idx].data_ptr() != args[idx].data_ptr():
                logger.warning_once(f"Mutated input[{arg_name}]'s data_ptr is different between capture and replay. "
                                    f"This may call redundant copy. ")
                args[idx].copy_(self._graphs_meta[fn_key].captured_inputs[idx])

        return self.reconstruct_outputs(fn_key)

    @property
    def config(self):
        return self._config

    @property
    def graph(self):
        return {graph_key: graph_meta.acl_graph for graph_key, graph_meta in self._graphs_meta.items()}

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
    def device(self):
        return self._device

    @property
    def fx_graph(self):
        return self._fx_graph

    @property
    def fx_graph_name(self):
        return self._graph_name

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
        multi_stream_enabled = apply_event_closure_with_multi_stream(self.fx_graph)
        logger.debug('after apply_stream_event_closure optimization, '
                     'multi_stream_enabled is %s, graph is %s.', multi_stream_enabled, self.fx_graph.graph)
        # graph optimization passes here
        # Note: this pass need sample args to run in FakeTensor mode, any pass modifies ops without meta registration
        # should run after it.

        if not (multi_stream_enabled or self.config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass):
            logger.debug("Start to process reinplace inplaceable ops fx pass for graph: %s", self.fx_graph_name)
            from torchair._acl_concrete_graph.graph_pass import _reinplace_inplaceable_ops_pass
            _reinplace_inplaceable_ops_pass(self.fx_graph, *sample_args)

        from torchair._acl_concrete_graph.acl_graph import replace_dynamic_workspace_ops, _find_mutated_user_inputs
        replace_dynamic_workspace_ops(self.fx_graph, self._meta_inputs)

        # Note: this will modify mutated input ops in fx graph, should be executed LAST.
        if not (multi_stream_enabled or self.config.debug.aclgraph.disable_reinplace_input_mutated_ops_pass):
            logger.debug("Start to process reinplace input mutated ops fx pass for graph: %s", self.fx_graph_name)
            from torchair._acl_concrete_graph.graph_pass import _reinplace_input_mutated_ops
            _reinplace_input_mutated_ops(self.fx_graph)

        # find mutated inputs after graph optimization passes
        self._mutated_user_inputs = _find_mutated_user_inputs(self.fx_graph)
        logger.debug('find mutated user inputs: %s', self._mutated_user_inputs)
        logger.debug('after graph optimization, graph is %s', self.fx_graph.graph)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path(f"dynamo_optimized_{self.fx_graph_name}"))

    def compile(self, *args: Any, **kwargs: Any):
        if not self._captured:
            # warm up before capture
            with record_function("acl_graph_warm_up"):
                for _ in range(self.num_warmup_iters):
                    self.fx_graph(*args, **kwargs)
                    torch.npu.synchronize()

            from torchair._acl_concrete_graph.acl_graph import get_unupdated_input_fn, get_updated_ops_fn
            self._unupdated_input_func = get_unupdated_input_fn(self.fx_graph)
            self._updated_input_func, self._updated_ops_param = get_updated_ops_fn(self.fx_graph, self._meta_inputs)
            self._captured = True

            # In the current version, the initialization of mem pool requires an explicit call to capture.
            # In versions greater than 2.6, the initialization can be completed directly when creating the mem pool.
            import torch_npu
            s = torch_npu.npu.Stream()
            with torch_npu.npu.stream(s):
                g = torch_npu.npu.NPUGraph()
                g.capture_begin(pool=self.pool)
                g.capture_end()
            # record the original memory state before capture,
            # and it will be used to restore the mem state when capturing another acl graph for different shape.
            self._original_mem_state = torch_npu._C._npu_getCheckpointState(self.device, self.pool)

        # get graph key based on unupdated sym input shape or value
        graph_key = self._unupdated_input_func(*args, **kwargs)
        if graph_key in self._graphs_meta.keys():
            logger.debug('Find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}.',
                         id(self.graph[graph_key]), self.fx_graph_name, graph_key)
            return graph_key

        # start capture aclgraph
        import torch_npu
        self._graphs_meta[graph_key] = GraphMeta(graph_key=graph_key,
                                                 acl_graph=torch_npu.npu.NPUGraph(),
                                                 replay_func=None,
                                                 captured_inputs=args,
                                                 outputs_meta=[],
                                                 outputs_weakref=[],
                                                 mem_state_after_capture=None,
                                                 is_first_replay=True)
        logger.debug('No find captured AclGraph{id: %s} of fx_graph %s with graph key {%s}, and start to capture it.',
                     id(self.graph[graph_key]), self.fx_graph_name, graph_key)

        stale_storage_set = set()
        for key, graph_meta in self._graphs_meta.items():
            if self._graphs_meta[key].outputs_weakref is None:
                continue
            for output_ref in self._graphs_meta[key].outputs_weakref:
                ref = output_ref()
                if ref is not None and isinstance(ref, torch.Tensor):
                    stale_storage_set.add(ref.untyped_storage()._cdata)
        stale_storages = list(stale_storage_set)
        torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state, stale_storages, [])
        logger.debug('After setting to original memory state for fx_graph %s for graph key{%s}. '
                     'The stale storage is %s, and the current memory state is {%s}.',
                     self.fx_graph_name, graph_key, stale_storages, debug_mem_state())

        with record_function("acl_graph_capture"):
            self.capture(graph_key, *args, **kwargs)

        return graph_key

    def capture(self, graph_key, *args: Any, **kwargs: Any):
        from torchair._acl_concrete_graph.acl_graph import (UpdatedNodeCaptureInterp, CapturedGraphUpdateAndReplay)
        captured_interpreter = UpdatedNodeCaptureInterp(self.fx_graph, self._updated_ops_param)

        import torch_npu
        with torch_npu.npu.graph(self.graph[graph_key], pool=self.pool, stream=self.stream,
                                 capture_error_mode=self.capture_error_mode):
            captured_outputs = captured_interpreter.run(*args, **kwargs)

        updated_node_infos = captured_interpreter.captured_node_infos
        logger.info('Success to capture fx_graph %s for graph key{%s}. '
                    'Start to run AclGraph{id: %s} with the updated node num {%s}.',
                    self.fx_graph_name, graph_key, id(self.graph[graph_key]), len(updated_node_infos))

        # The captured output tensors will not be held indefinitely,
        # and its will be terminated after the capture ends.
        import torch_npu
        self._graphs_meta[graph_key].mem_state_after_capture = \
            torch_npu._C._npu_getCheckpointState(self.device, self.pool)

        for output_iter in captured_outputs:
            if isinstance(output_iter, torch.Tensor):
                weak_ref = WeakRef(None)
            else:
                weak_ref = WeakRef(output_iter)
            self._graphs_meta[graph_key].outputs_weakref.append(weak_ref)
            self._graphs_meta[graph_key].outputs_meta.append(get_tensor_metadata(output_iter))
        del captured_outputs
        logger.debug('After capturing fx_graph %s for graph key{%s} to AclGraph{id: %s}, the memory state is {%s}.',
                     self.fx_graph_name, graph_key, id(self.graph[graph_key]), debug_mem_state())

        # gen run func
        self._graphs_meta[graph_key].replay_func = CapturedGraphUpdateAndReplay(self.graph[graph_key],
                                                                                self._updated_input_func,
                                                                                updated_node_infos)
        logger.debug('In graph {%s}, all the non parameter tensor input index list is: {%s}.',
                     self.fx_graph_name, self._user_inputs_mapping.values())

    def reconstruct_outputs(self, graph_key: str) -> List:
        """
        Reconstruct output tensors according to their saved metadata.
        Do not increase the reference count to the output tensors, and only weak reference is recorded.
        """

        if len(self._graphs_meta[graph_key].outputs_meta) != len(self._graphs_meta[graph_key].outputs_weakref):
            raise RuntimeError(
                f'The lengths of the outputs tensor meta {len(self._graphs_meta[graph_key].outputs_meta)} and '
                f'the outputs tensor ref {len(self._graphs_meta[graph_key].outputs_weakref)} do not match.')

        outputs = []
        have_invalid_weakref = False

        for idx, output_meta in enumerate(self._graphs_meta[graph_key].outputs_meta):
            output_ref = self._graphs_meta[graph_key].outputs_weakref[idx]()
            if output_ref is None:
                output_i = reconstruct_from_tensor_metadata(output_meta)
                self._graphs_meta[graph_key].outputs_weakref[idx].swap_weakref(output_i)
                outputs.append(output_i)
                have_invalid_weakref = True
            else:
                # valid tensor ref and other type obj can be returned directly.
                outputs.append(output_ref)

        if have_invalid_weakref:
            reconstructed_outputs_to_add_deleter = []
            for output_i in outputs:
                if isinstance(output_i, torch.Tensor):
                    reconstructed_outputs_to_add_deleter.append(output_i.untyped_storage()._cdata)

            other_graph_stale_storages = []
            stale_storage_set = set()
            for key, graph_meta in self._graphs_meta.items():
                if key == graph_key:
                    continue
                for output_ref in self._graphs_meta[key].outputs_weakref:
                    ref = output_ref()
                    if ref is not None and isinstance(ref, torch.Tensor):
                        stale_storage_set.add(ref.untyped_storage()._cdata)
            other_graph_stale_storages = list(stale_storage_set)

            import torch_npu
            if len(other_graph_stale_storages) > 0 and not self._graphs_meta[graph_key].is_first_replay:
                # reset other graph live tensors to stale storages
                torch_npu._C._npu_setCheckpointPoolState(self.device, self._original_mem_state,
                                                         other_graph_stale_storages, [])

            logger.debug('Reset fx_graph %s other graph key outputs stale storage cdata %s, '
                         'and set to original memory state for AclGraph{id: %s} with graph key{%s}.',
                         self.fx_graph_name, other_graph_stale_storages, id(self.graph[graph_key]), graph_key)

            # currently we deallocate on instead of allowing stale recordings
            stale_storages: List[int] = []
            torch_npu._C._npu_setCheckpointPoolState(self.device, self._graphs_meta[graph_key].mem_state_after_capture,
                                                     stale_storages, reconstructed_outputs_to_add_deleter)
            logger.debug('After reconstructing fx_graph %s graph key{%s} outputs, '
                         'the storages to add deleter are %s, the memory state is {%s}.',
                         self.fx_graph_name, graph_key, reconstructed_outputs_to_add_deleter, debug_mem_state())
        else:
            logger.debug('All output tensors weak ref are valid, '
                         'no need to reconstruct fx_graph %s for graph key{%s}.',
                         self.fx_graph_name, graph_key)

        if self._graphs_meta[graph_key].is_first_replay:
            self._graphs_meta[graph_key].is_first_replay = False
        return outputs

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

        return meta_outputs
