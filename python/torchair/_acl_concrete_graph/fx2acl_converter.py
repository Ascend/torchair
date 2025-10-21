from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import functools
import logging
import sys
import warnings
import sympy

import torch
from torch import fx
from torch._subclasses.fake_tensor import is_fake
from torch.fx.node import Argument, Target
from torch.profiler import record_function
from torch.types import Device, Number
from packaging import version

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
    """
    AclConcreteGraph represents a concrete computation graph optimized for Ascend NPU devices.
    It extends the base ConcreteGraphBase to provide ACL-specific compilation and execution capabilities.
    
    Args:
        config (CompilerConfig): Configuration object for compiler settings.
        name (str, optional): Name of the graph. Defaults to "graph".
        pool (Optional[Any], optional): Memory pool handle for ACL operations. Defaults to None.
        stream (Optional[Any], optional): Execution stream for asynchronous operations. Defaults to None.
        capture_error_mode (str, optional): Error handling mode during graph capture. Defaults to "global".
        num_warmup_iters (int, optional): Number of warm-up iterations before capturing the graph. Defaults to 0.
    """

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
        self._fx_input_names = []
        self._all_sym_input_idx = {}
        self._all_meta_tensor_input = {}
        self._fx_graph: fx.GraphModule = None
        self._fx_forward: str = None
        self._aclgraph_manager: AclGraph = None
        self._aclgraph_cache_info = AclGraphCacheInfo(
            pool=pool,
            stream=stream,
            capture_error_mode=capture_error_mode,
            num_warmup_iters=num_warmup_iters,
            fx_graph_name=name,
            user_inputs_mapping=OrderedDict(),
            parameter_user_inputs=[]
        )
        self._tensor_constant_dict = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the compiled ACL graph with the provided inputs.
        
        This method handles input processing, graph execution, and output retrieval.
        It ensures proper data synchronization between captured inputs and user-provided inputs
        for in-place operations that may modify tensor addresses.
        
        Args:
            *args: Variable length argument list for graph inputs.
            **kwargs: Arbitrary keyword arguments for graph inputs.
        
        Returns:
            Any: Output tensors from the executed graph.
        """

        # get graph_key and capture
        fn_key = self.compile(*args, **kwargs)

        # fall back to eager when static_capture_size_limit is exceeded
        if self.graph.fallback_to_eager:
            with record_function("fx_run_eagerly"):
                return self.graph.fx_run_eagerly(*args, **kwargs)

        # input process
        self.graph.process_input(fn_key, *args)

        # run/replay
        with record_function("acl_graph_replay"):
            self.graph.run(fn_key, *args, **kwargs)

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
    def fx_forward(self):
        return self._fx_forward

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
        if len(self._tensor_constant_dict) > 0:
            tensor_const_code = self._codegen_tensor_constant()
            head.splice(tensor_const_code)
        head.writeline('')
        forward_code = self.fx_forward
        head.splice(forward_code)
        head.writeline('')
        init_code = self._codegen_init()
        head.splice(init_code)
        need_update_tagged_event = (len(self._aclgraph_cache_info.tagged_event_names) > 0)
        if need_update_tagged_event:
            head.writeline('')
            update_code = self._codegen_update_tagged_event()
            head.splice(update_code)
        head.writeline('')
        kernel_code = self._codegen_kernel(need_update_tagged_event)
        head.splice(kernel_code)

        return head.getvalue()

    def compile(self, *args: Any, **kwargs: Any):
        """
        Compiles the computation graph into an executable ACL graph.
        
        This method performs graph capture, optimization, and key generation for subsequent executions.
        
        Args:
            *args: Input arguments for graph compilation.
            **kwargs: Keyword arguments for graph compilation.
            
        Returns:
            str: Unique identifier (graph key) for the captured ACL graph.
        """
        return self.graph.compile(*args, **kwargs)

    def optimize_graph_without_runtime(self, *sample_args):
        """
        Optimizes the computation graph without relying on runtime information.
        This includes passes like re-inplacing in-place operations and dynamic workspace handling.
        
        Args:
            *sample_args: Sample input arguments for tracing and optimization.
        """

        logger.debug('before graph optimization, graph is %s', self.fx_graph.graph)
        if self.config.aclgraph_config.use_custom_pool is not None:
            # when use custom pool from user, do not enable mem pool reuse in same fx.
            self.config.debug.aclgraph.disable_mempool_reuse_in_same_fx = True

        multi_stream_enabled = apply_event_closure_with_multi_stream(self.fx_graph,
                                                                     self.fx_graph_name,
                                                                     self._aclgraph_cache_info.tagged_event_names)
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
        # It is necessary to recompile GraphModule to make sure that changes to graph take effect.
        self.fx_graph.recompile()
        self._fx_forward = self._codegen_fx_forward(self.fx_graph, self.fx_graph.code,
                                                    self._aclgraph_cache_info.updated_ops_param)
        logger.debug('Original fx_forward is: %s', self.fx_graph.code)
        self._aclgraph_manager = AclGraph(fx_graph=self.fx_graph, config=configs)
        self.graph.load(self._aclgraph_cache_info)

    def normalize_config(self):
        aclgraph_config_options = self.config.debug.aclgraph.as_dict()
        aclnn_static_shape_kernel = self.config.experimental_config.aclgraph._aclnn_static_shape_kernel
        if aclnn_static_shape_kernel:
            aclgraph_config_options['_aclnn_static_shape_kernel'] = aclnn_static_shape_kernel.value
        build_dir = self.config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir.value
        if build_dir:
            aclgraph_config_options['_aclnn_static_shape_kernel_build_dir'] = build_dir
        frozen_parameter_enable = self.config.experimental_config.frozen_parameter
        if frozen_parameter_enable:
            if version.parse(torch.__version__) < version.parse("2.5.1"):
                warnings.warn('When enable frozen_parameter, Parameters will be considered static. '
                              'Please make sure that the Parameters data address remain the same '
                              'throughout the program runtime.')
            else:
                warnings.warn('When enable frozen_parameter, Parameters and input tensors with immutable data_ptr '
                              'marked by `torch._dynamo.mark_static_address()` will be considered static. '
                              'Please make sure that the Parameters data address remain the same '
                              'throughout the program runtime.')
            aclgraph_config_options['frozen_parameter'] = frozen_parameter_enable.value

        logger.debug("aclgraph compile options:")
        for k, v in aclgraph_config_options.items():
            logger.debug("  %s: %s", k, v)

        return aclgraph_config_options

    def parse_symlist(self, syms):
        """
        Parses a list of symbols (either integers or ValuePack objects) into a list of NPU-compatible symbols.
        
        Args:
            syms (List[Union[int, ValuePack]]): List containing symbols to parse.
            
        Returns:
            List[int]: Parsed list of integer symbols.
        """
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
        """
        Parses input metadata during graph construction.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
            
        Returns:
            Any: Processed metadata for the input operation.
        """
        self._meta_inputs.append(meta_outputs)

        # Lazy check for int/sym inputs
        if isinstance(meta_outputs, torch.Tensor):
            if isinstance(meta_outputs, torch.nn.Parameter) or hasattr(meta_outputs, "_dynamo_static_input_type"):
                self._aclgraph_cache_info.parameter_user_inputs.append(len(self._meta_inputs) - 1)
            else:
                self._aclgraph_cache_info.user_inputs_mapping.setdefault(target, len(self._meta_inputs) - 1)
        # for assert_size_stride
        self._fx_input_names.append(target)
        if is_sym(meta_outputs):
            self._all_sym_input_idx[meta_outputs.node.expr] = len(self._meta_inputs) - 1
        else:
            self._all_meta_tensor_input[len(self._meta_inputs) - 1] = meta_outputs
        return meta_outputs

    def parse_node(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        """
        Parses individual nodes within the graph during compilation.
        
        This method can be extended to include optimizations specific to certain node types.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
            
        Returns:
            Any: Processed result of the parsed node.
        """
        # do some optimization in fx for some ops

        return target(*args, **kwargs)

    def parse_output(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], meta_outputs: Any):
        """
        Parses output metadata during graph construction.
        
        Args:
            target (Target): The target operation being parsed.
            args (Tuple[Argument, ...]): Input arguments for the target operation.
            kwargs (Dict[str, Any]): Keyword arguments for the target operation.
            meta_outputs (Any): Metadata associated with the operation's outputs.
            
        Returns:
            Any: Processed metadata for the output operation.
        """
        if not (isinstance(args, (list, tuple)) and len(args) == 1):
            raise RuntimeError(f"Unsupported case in AclGraph: for output node with args: [{args}].")

        return meta_outputs

    def _codegen_tensor_constant(self):
        from torch._inductor.utils import IndentedBuffer
        tensor_constant_code = IndentedBuffer()
        tensor_constants_list = []
        # We must ensure that 'self._tensor_constant_dict[k].cpu().item()' is executed
        # outside any FakeTensorMode context.
        with torch._C._DisableTorchDispatch():
            for k, v in self._tensor_constant_dict.items():
                if not isinstance(v, torch.Tensor):
                    raise TypeError(f"When iterating self._tensor_constant_dict, "
                                    f"Value for key '{k}' is not a tensor, got {type(v)}")
                try:
                    tensor_list = (v.cpu().item() if v.dim() == 0 else v.cpu().tolist())
                    tensor_constants_list.append(f"tensor_constants['{k}'] = "
                                                 f"torch.tensor({tensor_list}, "
                                                 f"device='{self._tensor_constant_dict[k].device}')."
                                                 f"reshape({self._tensor_constant_dict[k].shape})")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to generate tensor constant for key '{k}'. "
                        f"Error: {str(e)}"
                    ) from e
        tensor_constant_code.writelines(["", "tensor_constants = {}", "with torch._C._DisableTorchDispatch():"])
        with tensor_constant_code.indent():
            tensor_constant_code.writelines(tensor_constants_list)
        tensor_constant_code.writeline("")
        return tensor_constant_code.getvalue()

    def _codegen_init(self):
        from torch._inductor.utils import IndentedBuffer
        init_code = IndentedBuffer()
        # Please make sure that fx graph will not be changed/optimized after generate fx_forward

        init_code.writelines(['',
                              f'compile_configs = {{}}'])
        configs = self.normalize_config()
        for k, v in configs.items():
            init_code.writeline(f'compile_configs["{k}"] = "{v}"')

        init_code.writelines(['',
                              f'acl_graph = AclGraph(fx_forward=forward, '
                              f'config=compile_configs)'])

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
                mutated_user_inputs={self._aclgraph_cache_info.mutated_user_inputs},
                tagged_event_names={self._aclgraph_cache_info.tagged_event_names},
                parameter_user_inputs={self._aclgraph_cache_info.parameter_user_inputs}
            )
            acl_graph.load(aclgraph_cache_info)
        ''')

        return init_code.getvalue()

    def _codegen_update_tagged_event(self):
        from torch._inductor.utils import IndentedBuffer
        update_code = IndentedBuffer()
        update_code.splice('''
        def _update_tagged_event_dict():
            from torchair._acl_concrete_graph.graph_pass import _GLOBAL_SCOPE_TAG_TO_EVENT, _GLOBAL_EVENT_LOCK
            with _GLOBAL_EVENT_LOCK:
                for i, tag in enumerate(aclgraph_cache_info.tagged_event_names):
                    tagged_event = torch.npu.Event()
                    _GLOBAL_SCOPE_TAG_TO_EVENT[tag] = tagged_event
        ''')
        return update_code.getvalue()

    def _codegen_kernel(self, need_update_tagged_event=False):
        from torch._inductor.utils import IndentedBuffer
        kernel_code = IndentedBuffer()
        kernel_code.writelines(['', '_is_first_run = True', f'def kernel(*args, **kwargs):'])
        with kernel_code.indent():
            # for assert_shape_stride in first run
            kernel_code.writelines(['', 'global _is_first_run', 'if _is_first_run:'])
            with kernel_code.indent():
                kernel_code.writelines(['_is_first_run = False', ''])
                if need_update_tagged_event:
                    kernel_code.writelines(['_update_tagged_event_dict()', ''])
                input_code = self._codegen_input()
                kernel_code.splice(input_code)
                assert_code = self._codegen_assert_size_stride()
                kernel_code.splice(assert_code)
            kernel_code.splice('''
                    fn_key = acl_graph.compile(*args)

                    if acl_graph.fallback_to_eager:
                        with record_function("fx_run_eagerly"):
                            return acl_graph.fx_run_eagerly(*args, **kwargs)

                    acl_graph.process_input(fn_key, *args)

                    with record_function("acl_graph_replay"):
                        acl_graph.run(fn_key, *args)

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

    def _codegen_fx_forward(self, gm: torch.fx.GraphModule, code: str, need_updated_ops: Dict):
        for node in gm.graph.nodes:
            # Only record tensor_constants for now.
            if node.op == "get_attr" and "tensor_constant" in node.name:
                self._tensor_constant_dict[node.name] = getattr(gm, node.target)
        import re
        forward_def_match = re.search(r"def forward\(self[^)]*\):", code)
        if not forward_def_match:
            raise ValueError("Cannot find 'forward' in the code which is generated from recompile of a GraphModule.")

        # get func body and split it by line
        body_start = forward_def_match.end()
        body = code[body_start:].splitlines()

        from torch._inductor.utils import IndentedBuffer
        forward_code = IndentedBuffer()
        # func signiture
        forward_code.writeline("def forward(*args, node_info=[], is_capturing: bool = False):")

        with forward_code.indent():
            all_input_str = ', '.join(self._fx_input_names)
            if all_input_str:
                if len(self._fx_input_names) == 1:
                    all_input_str += ', '
            forward_code.writeline(f'{all_input_str} = args')
            need_updated_ops_dict = {}
            has_need_updated_ops = self._codegen_fx_forward_updated_ops(gm, need_updated_ops, need_updated_ops_dict)
            if has_need_updated_ops:
                forward_code.writelines(["from torchair._acl_concrete_graph.utils import reconstruct_args_kwargs",
                                         "from torchair._acl_concrete_graph.acl_graph import UpdatedNodeInfo"])
            for line in body:
                need_update = False
                for k in need_updated_ops_dict.keys():
                    if k in line:
                        forward_code.splice(need_updated_ops_dict[k])
                        # Frees memory that won't be used afterward.
                        line_parts = line.split(';', 1)
                        mem_free_part = line_parts[1].strip() if len(line_parts) > 1 else ""
                        forward_code.writeline(mem_free_part)
                        need_update = True
                        break
                for k in self._tensor_constant_dict.keys():
                    if f"{k} = self" in line:
                        forward_code.writeline(f"{k} = tensor_constants['{k}']")
                        need_update = True
                        break
                if not need_update:
                    forward_code.writeline(line.strip())

        return forward_code.getvalue()

    def _codegen_fx_forward_updated_ops(self, gm: torch.fx.GraphModule, need_updated_ops: Dict,
                                        need_updated_ops_dict: Dict):
        has_need_updated_ops = False
        from torch._inductor.utils import IndentedBuffer
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.name in need_updated_ops.keys():
                need_updated_ops_code = IndentedBuffer()
                has_need_updated_ops = True

                # external event no need record before capture.
                need_updated_ops_code.splice(f'''
                if is_capturing:
                    external_event_{node.name} = torch.npu.ExternalEvent()
                    capture_stream_{node.name} = torch.npu.current_stream()
                    external_event_{node.name}.wait(capture_stream_{node.name})
                    external_event_{node.name}.reset(capture_stream_{node.name})
                    torch.npu.graph_task_group_begin(capture_stream_{node.name})

                {node.name} = torch.ops.{node.target}(*{node.args}, **{node.kwargs})

                if is_capturing:
                    handle_{node.name} = torch.npu.graph_task_group_end(capture_stream_{node.name})
                    node_args, node_kwargs = reconstruct_args_kwargs({node.args}, {node.kwargs})
                    node_info.append(UpdatedNodeInfo(
                        node_name="{node.name}",
                        updated_func=torch.ops.{node.target},
                        updated_param_name={need_updated_ops[node.name]},
                        args=node_args,
                        kwargs=node_kwargs,
                        handle=handle_{node.name},
                        event=external_event_{node.name})
                    )

                ''')
                need_updated_ops_dict[f'{node.name} = torch.ops.{node.target}'] = \
                    need_updated_ops_code.getvalue()

        return has_need_updated_ops
