from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import functools
import logging
import sys
import sympy

import torch
from torch import fx
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
from torchair._utils.path_manager import PathManager
from torchair._acl_concrete_graph.acl_graph import AclGraph, AclGraphCacheInfo, is_sym
from torchair._acl_concrete_graph.graph_pass import apply_event_closure_with_multi_stream

aten = torch.ops.aten


class AclConcreteGraph(ConcreteGraphBase):
    def __init__(self, config: CompilerConfig, name="graph", pool=None, stream=None,
                 capture_error_mode: str = "global", num_warmup_iters=0):
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                "Couldn't import torch_npu. When the CompilerConfig.mode is reduce-overhead, "
                "it is necessary to use torch_npu.npu.NPUGraph(), so importing torch_npu is essential.") from e
        # for AclConcreteGraph only
        self._config = config
        self._meta_inputs = []
        self._meta_outputs = []
        self._fx_input_names = []
        self._all_sym_input_idx = {}
        self._all_meta_tensor_input = {}
        self._fx_graph: fx.GraphModule = None
        self._aclgraph_manager: AclGraph = None
        self._aclgraph_cache_info = AclGraphCacheInfo(
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
            num_warmup_iters=num_warmup_iters,
            fx_graph_name=name,
            user_inputs_mapping=OrderedDict()
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # get graph_key and capture
        fn_key = self.compile(*args, **kwargs)

        # input process
        self.graph.process_input(fn_key, *args)

        # run/replay
        with record_function("acl_graph_replay"):
            self.graph.run(fn_key, *args, **kwargs)

        # For in-place op, dynamo will transform it into a functionalized call and add copy_ node when setting
        # keep_inference_input_mutations=True, which may need data copy from capture input to user input (when tensor
        # address is different between capture and replay).
        self.graph.process_inplace_inputs(fn_key, *args)

        return self.graph.reconstruct_outputs(fn_key)

    @property
    def config(self):
        return self._config

    @property
    def graph(self):
        return self._aclgraph_manager

    @property
    def fx_graph(self):
        return self._fx_graph

    @property
    def fx_graph_name(self):
        return self._aclgraph_cache_info.fx_graph_name

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
        from torch._inductor.utils import IndentedBuffer
        head = IndentedBuffer()
        # DONT CHANGE class names of AclGraph and AclGraphCacheInfo
        # DONT CHANGE import path of "AclGraph, AclGraphCacheInfo"
        # which is "from torchair._acl_concrete_graph.acl_graph import"
        head.splice('''
            from collections import OrderedDict
            from typing import List, Optional, Callable, Any, Dict, Tuple, Union
            import torch
            from torch.profiler import record_function
            import torch_npu
            from torchair._acl_concrete_graph.acl_graph import AclGraph, AclGraphCacheInfo
            from torchair.configs.compiler_config import CompilerConfig
            assert_size_stride = torch._C._dynamo.guards.assert_size_stride
            ''')

        # There is no need to save config.for now. Maybe needed in the future.
        # Configs are set for AclConcreteGraph, not for AclGraph
        init_code = self._codegen_init()
        head.splice(init_code)

        kernel_code = self._codegen_kernel()
        head.splice(kernel_code)

        return head.getvalue()

    def compile(self, *args: Any, **kwargs: Any):
        return self.graph.compile(*args, **kwargs)

    def optimize_graph_without_runtime(self, *sample_args):
        logger.debug('before graph optimization, graph is %s', self.fx_graph.graph)
        if self.config.aclgraph_config.use_custom_pool is not None:
            # when use custom pool from user, do not enable mem pool reuse in same fx.
            self.config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True

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
        self._aclgraph_cache_info.mutated_user_inputs = _find_mutated_user_inputs(self.fx_graph)
        logger.debug('find mutated user inputs: %s', self._aclgraph_cache_info.mutated_user_inputs)
        logger.debug('after graph optimization, graph is %s', self.fx_graph.graph)

        if self.config.debug.graph_dump.enabled:
            self.dump(self.config.debug.graph_dump.full_path(f"dynamo_optimized_{self.fx_graph_name}"))

        # get info for get_unupdated_input_fn and get_updated_ops_fn from fx_graph
        from torchair._acl_concrete_graph.acl_graph import get_unupdated_sym_input_index, get_updated_ops_rulers_param
        self._aclgraph_cache_info.unupdated_sym_input_index = get_unupdated_sym_input_index(self.fx_graph)
        self._aclgraph_cache_info.ops_update_rulers, self._aclgraph_cache_info.updated_ops_param = \
            get_updated_ops_rulers_param(self.fx_graph, self._meta_inputs)

        # Must not optimize fx_graph after this. Initialize aclgraph.
        configs = self.normalize_config()
        self._aclgraph_manager = AclGraph(fx_graph=self.fx_graph, config=configs)
        self.graph.load(self._aclgraph_cache_info)

    def normalize_config(self):
        aclgraph_config_options = self.config.debug.aclgraph.as_dict()

        logger.debug("aclgraph compile options:")
        for k, v in aclgraph_config_options.items():
            logger.debug("  %s: %s", k, v)

        return aclgraph_config_options

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
                self._aclgraph_cache_info.user_inputs_mapping.setdefault(target, len(self._meta_inputs) - 1)
        # for assert_size_stride
        self._fx_input_names.append(target)
        if is_sym(meta_outputs):
            self._all_sym_input_idx[meta_outputs.node.expr] = len(self._meta_inputs) - 1
        else:
            self._all_meta_tensor_input[len(self._meta_inputs) - 1] = meta_outputs
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

    def _codegen_init(self):
        from torch._inductor.utils import IndentedBuffer
        init_code = IndentedBuffer()
        # Please make sure that fx graph will not be changed/optimized after serialized_fx_graph

        init_code.writelines(['',
                              f'serialized_fx_graph = {AclGraph.save_graphmodule_to_str(self.fx_graph)}',
                              f'compile_configs = {{}}'])
        configs = self.normalize_config()
        for k, v in configs.items():
            init_code.writeline(f'compile_configs["{k}"] = "{v}"')

        init_code.writelines(['',
                              f'acl_graph = AclGraph(serialized_fx_graph=serialized_fx_graph, config=compile_configs)'])

        init_code.splice(f'''
            aclgraph_cache_info = AclGraphCacheInfo(
                pool={self._aclgraph_cache_info.pool},
                stream={self._aclgraph_cache_info.stream},
                capture_error_mode="{self._aclgraph_cache_info.capture_error_mode}",
                num_warmup_iters={self._aclgraph_cache_info.num_warmup_iters},
                fx_graph_name="{self._aclgraph_cache_info.fx_graph_name}",
                user_inputs_mapping={self._aclgraph_cache_info.user_inputs_mapping},
                unupdated_sym_input_index={self._aclgraph_cache_info.unupdated_sym_input_index},
                updated_ops_param={self._aclgraph_cache_info.updated_ops_param},
                ops_update_rulers={self._aclgraph_cache_info.ops_update_rulers},
                mutated_user_inputs={self._aclgraph_cache_info.mutated_user_inputs}
            )
            acl_graph.load(aclgraph_cache_info)
        ''')

        return init_code.getvalue()

    def _codegen_kernel(self):
        from torch._inductor.utils import IndentedBuffer
        kernel_code = IndentedBuffer()
        kernel_code.writelines(['', '_is_first_run = True', f'def kernel(*args):'])
        with kernel_code.indent():
            # for assert_shape_stride in first run
            kernel_code.writelines(['', 'global _is_first_run', 'if _is_first_run:'])
            with kernel_code.indent():
                kernel_code.writelines(['_is_first_run = False'])
                input_code = self._codegen_input()
                kernel_code.splice(input_code)
                assert_code = self._codegen_assert_size_stride()
                kernel_code.splice(assert_code)
            kernel_code.splice('''
                    fn_key = acl_graph.compile(*args)

                    acl_graph.process_input(fn_key, *args)

                    with record_function("acl_graph_replay"):
                        acl_graph.run(fn_key, *args)

                    acl_graph.process_inplace_inputs(fn_key, *args)

                    return acl_graph.reconstruct_outputs(fn_key)

                ''')
        return kernel_code.getvalue()

    def _codegen_assert_size_stride(self):
        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()

        for idx, meta in self._all_meta_tensor_input.items():
            input_code.writelines([f'assert_size_stride(args[{idx}], {tuple(meta.shape)}, {meta.stride()})'])

        return input_code.getvalue()

    def _codegen_input(self):
        from torch._inductor.utils import IndentedBuffer
        input_code = IndentedBuffer()
        if self._all_sym_input_idx:
            # only for cache with symint
            all_input_str = ', '.join(self._fx_input_names)
            if all_input_str:
                if len(self._fx_input_names) == 1:
                    all_input_str += ', '
            input_code.writeline(f'{all_input_str} = args')
            for name, idx in self._all_sym_input_idx.items():
                if str(name).isdigit() or not isinstance(name, sympy.Symbol):
                    continue
                input_code.writeline(f'{str(name)} = {self._fx_input_names[idx]}')
        return input_code.getvalue()
