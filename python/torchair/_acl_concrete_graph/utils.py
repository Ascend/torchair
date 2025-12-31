from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
import time
from typing import List, Optional, Callable, Any, Dict, Tuple, Union, Set
import weakref
import sys
import operator

import torch
from torch.types import Device, Number
from torch.fx import Graph, GraphModule, Node

from torchair.core.utils import logger
from torchair.ge._ge_graph import Format


class WeakRef:
    """
    Wrapper around a weak ref of a tensor or a direct obj.
    """

    def __init__(self, input):
        if isinstance(input, torch.Tensor):
            self._ref = weakref.ref(input)
            self._is_tensor = True
        else:
            self._ref = input
            self._is_tensor = False

    def __call__(self):
        if self._is_tensor:
            out = self._ref()
            # the returned obj out may be None when original obj is released.
            return out
        else:
            return self._ref

    def swap_weakref(self, input: Any):
        if isinstance(input, torch.Tensor):
            self._ref = weakref.ref(input)
            self._is_tensor = True
        else:
            self._ref = input
            self._is_tensor = False


class LazyMessage:
    """
    The LazyMessage class is designed to delay the execution of a function
    and obtain its string representation when needed.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.func(*self.args, **self.kwargs))


@dataclass
class TensorMetadata:
    """
    This represents all meta info for a tensor object.
    We can directly create a tensor based on the following information,
    but this tensor will do nothing when it is released.
    """

    size: torch.Size
    stride: List[int]
    nbytes: int
    dtype: torch.dtype
    data_ptr: Any
    storage_offset: int
    device: Device
    npu_format: int
    layout: Optional[torch.layout] = torch.strided
    memory_format: Optional[torch.memory_format] = torch.contiguous_format
    requires_grad: Optional[bool] = False


@dataclass
class GraphMeta:
    """
    This represents all meta info for a acl graph object.
    """

    graph_key: str
    acl_graph: Any
    replay_func: Callable
    outputs_meta: List[Union[TensorMetadata, Number]]
    outputs_weakref: List[WeakRef]
    mem_state_after_capture: Any
    userinputs_meta: Dict[int, Union[TensorMetadata, Number]] = field(default_factory=dict)
    userinputs_metatensor: Dict[int, Union[TensorMetadata, torch.Tensor]] = field(default_factory=dict)
    userinputs_weakref: Dict[int, WeakRef] = field(default_factory=dict)
    captured_parameter: Dict[int, int] = field(default_factory=dict)
    captured_mutated_inputs: Dict[int, int] = field(default_factory=dict)
    captured_userinput_ref_with_output: Dict[int, int] = field(default_factory=dict)
    retained_userinputs: Dict[int, Any] = field(default_factory=dict)
    captured_output_idx_ref_with_userinput: Set[int] = field(default_factory=set)
    retained_outputs: List[torch.Tensor] = None


def get_tensor_metadata(x):
    """
    Just record meta data for a pytorch tensor object.
    """
    if isinstance(x, torch.Tensor):
        npu_format = Format.FORMAT_ND.value
        torch_npu_module = sys.modules.get('torch_npu', None)
        if torch_npu_module is None:
            logger.info(f'The internal format will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, set format tensormetadata to FORMAT_ND.')
        else:
            if x.is_npu:
                npu_format = torch_npu_module.get_npu_format(x)

        return TensorMetadata(
            size=x.shape,
            stride=x.stride(),
            nbytes=x.untyped_storage().nbytes(),
            dtype=x.dtype,
            data_ptr=x.untyped_storage().data_ptr(),
            storage_offset=x.storage_offset(),
            device=x.device,
            npu_format=npu_format,
            layout=x.layout,
            requires_grad=x.requires_grad,
        )
    else:
        return x


def reconstruct_from_tensor_metadata(metadata: Dict[str, Any]) -> torch.Tensor:
    if not isinstance(metadata, TensorMetadata):
        raise RuntimeError(f"Unsupported input type[{type(metadata)}] "
                           f"when reconstructing tensor from metadata, expected input type: TensorMetadata.")

    metadata = asdict(metadata)
    import torch_npu
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )

    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)


def reconstruct_tensor_list(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    if not isinstance(tensor_list, (tuple, list)):
        raise RuntimeError(f"Unsupported input type[{type(tensor_list)}], expected input type: List or Tuple.")

    res_tensors = []
    for tensor_i in tensor_list:
        if isinstance(tensor_i, torch.Tensor):
            res_tensors.append(reconstruct_from_tensor_metadata(get_tensor_metadata(tensor_i)))
        else:
            res_tensors.append(tensor_i)
    return res_tensors


def reconstruct_args_kwargs(node_args, node_kwargs) -> Tuple[Any, Any]:
    ret_args = []
    for input_i in node_args:
        if isinstance(input_i, torch.Tensor):
            ret_args.append(reconstruct_from_tensor_metadata(get_tensor_metadata(input_i)))
        elif isinstance(input_i, (tuple, list)):
            ret_args.append(reconstruct_tensor_list(input_i))
        else:
            ret_args.append(input_i)

    ret_kwargs = {}
    for kk, vv in node_kwargs.items():
        if isinstance(vv, torch.Tensor):
            ret_kwargs[kk] = reconstruct_from_tensor_metadata(get_tensor_metadata(vv))
        elif isinstance(vv, (tuple, list)):
            ret_kwargs[kk] = reconstruct_tensor_list(vv)
        else:
            ret_kwargs[kk] = vv

    return ret_args, ret_kwargs


def debug_mem_state() -> str:
    segments = torch.npu.memory_snapshot()
    seg = []
    for segment in segments:
        if "segment_pool_id" in segment:
            if segment["segment_pool_id"] == (0, 0):
                continue

            tmp = ({"device": segment["device"]},
                   {"stream": segment["stream"]},
                   {"pool_id": segment["segment_pool_id"]},
                   {"block_num": len(segment["blocks"])},
                   {"activate_num": sum(int(blk["state"] == "active_allocated") for blk in segment["blocks"])},
                   {"total_size": segment["total_size"]},
                   {"allocated_size": segment["allocated_size"]},)
            seg.append(tmp)
    seg_str = "\n".join([str(seg_iter) for seg_iter in seg])
    return "\n" + seg_str


@contextmanager
def timer(prefix: str):
    start_time = time.time()
    yield
    logger.info("%s took %.6f [s]", prefix, time.time() - start_time)


_BASE_FORMAT_GET = {
    Format.FORMAT_NCHW.value,
    Format.FORMAT_NHWC.value,
    Format.FORMAT_ND.value,
    Format.FORMAT_NCDHW.value
}


def is_inputs_base_format(tensor_list: List[torch.Tensor]) -> bool:
    for tensor_i in tensor_list:
        if not is_op_input_base_format(tensor_i):
            return False
    return True


def is_op_input_base_format(tensor) -> bool:
    if tensor.is_cpu:
        return True
    if 'torch_npu' not in sys.modules:
        logger.info(f'The internal format will only be enabled in a torch npu env.'
                    'When there is no torch_npu in the env, skip internal format check.')
        return False
    torch_npu_module = sys.modules['torch_npu']
    npu_format = torch_npu_module.get_npu_format(tensor)
    if npu_format not in _BASE_FORMAT_GET:
        return False
    return True


def _is_copy_node(node: Node) -> bool:
    """Check if a node is a copy_ node."""
    return (
        node.op == "call_function" 
        and node.target == torch.ops.aten.copy_.default
    )


def get_inplace_op_mutated_input_users(
    original_node: Union[Node, List[Node]],
    mutated_input_indices: Union[Node, List[int]],
) -> Union[List[Node], List[List[Node]]]:
    """
    Returns all downstream users of the mutated input nodes from an in-place operation.

    For each specified input index that is mutated by the in-place operation,
    collects all subsequent nodes in the graph that consume the mutated value
    (including indirect users through intermediate nodes).

    Args:
        original_node: The original functionalized node that will be reinplaced,
                              or the mutated argument node for auto_functionalize.
        mutated_input_indices: List of indices pointing to which inputs will be mutated.
                               If None, original_node is treated as a single node
                               and its users are returned.

    Returns:
        List of user node lists if mutated_input_indices is provided,
        otherwise returns a single list of user nodes.
    """
    # Handle single node case (for auto_functionalize)
    if isinstance(mutated_input_indices, Node):
        return _collect_single_node_users(mutated_input_indices)
    
    # Handle multiple nodes case (for regular reinplace checks)
    all_inputs_users = []
    
    for idx in mutated_input_indices:
        if idx >= len(original_node.args):
            logger.warning("Input index %d out of range for node %s", idx, original_node.name)
            continue
            
        input_node = original_node.args[idx]
        users = _collect_single_node_users(input_node)
        all_inputs_users.append(users)
    
    return all_inputs_users


def _collect_single_node_users(input_node: Node) -> List[Node]:
    """
    Collects all subsequent nodes in the graph that consume the mutated value
    (including indirect users through intermediate nodes).
    """
    users = []
    copy_node = None
    
    # part 1: all users of input node
    # Collect non-copy users of the input node
    for user in input_node.users:
        if _is_copy_node(user):
            # user copy_ will be replace by input
            copy_node = user
            continue
        users.append(user)
    
    # part 2: all users of functionalized op output
    # Only when multi reinplace is need,
    # because functionalized op output(get_item) will be replace by input only in multi reinplace case.
    if copy_node is not None:
        # Collect non-copy users of the functionalized output
        if len(copy_node.args) > 1 and copy_node.args[1].target == operator.getitem:
            getitem_node = copy_node.args[1]
            for candidate in getitem_node.users:
                if candidate != copy_node:
                    # user copy_ will be replace by input
                    users.append(candidate)
        
        # part 3: all users of copy_. Collect users of the copy node itself
        users.extend(copy_node.users)
    
    return users


class ReinplaceStreamChecker:
    """Unified handler for reinplace operation stream checking."""
    
    def __init__(self):
        pass

    def check_single_reinplace(self, node: torch.fx.Node) -> bool:
        """Check stream consistency for single-input reinplace operations."""
        return self._check_reinplace_streams(node, [0])
    
    def check_multi_reinplace(self, node: torch.fx.Node) -> bool:
        """Check stream consistency for multi-input reinplace operations."""
        from torchair._acl_concrete_graph.graph_pass import inplaceable_npu_ops
        inplace_op = inplaceable_npu_ops.get(node.target)
        if inplace_op is None:
            return False
        return self._check_reinplace_streams(node, inplace_op.mutated_arg)
    
    def check_auto_functionalize(self, node: torch.fx.Node, mutated_arg: torch.fx.Node) -> bool:
        """Check stream consistency for auto_functionalize scenarios."""
        # Use unified user collection interface
        users = get_inplace_op_mutated_input_users(node, mutated_arg)
        return self._verify_and_log(node, users, "multi_stream_auto_functionalize")
    
    def _check_reinplace_streams(self, node: torch.fx.Node, input_indices: List[int]) -> bool:
        """Internal method: Check stream consistency for specified input indices."""
        all_inputs_users = get_inplace_op_mutated_input_users(node, input_indices)
        check_name = "multi_stream_single_reinplace" if len(input_indices) == 1 else "multi_stream_multi_reinplace"
        logger.debug(f"[Reinplace check_reinplace_streams]Check Node:{node.name} "
                     f"with inplace args indices:{input_indices}, which are {[node.args[i] for i in input_indices]} "
                     f"has all_inputs_users:{all_inputs_users} respectively.")
        for users in all_inputs_users:
            if not self._verify_and_log(node, users, check_name):
                return False
        return True
    
    def _verify_and_log(self, node: Node, users: List[Node], check_name: str) -> bool:
        """Verify if user nodes are on the same stream and log the result."""
        from torchair._utils.graph_utils import verify_nodes_on_same_stream
        
        if verify_nodes_on_same_stream(users):
            logger.debug("[%s]Current node: %s, type: %s check no multi stream users success for reinplace. "
                 "The users of the mutated input node did not have multiple streams.", check_name,
                 node.name, node.target)
            return True
        else:
            logger.debug("[%s]Current node: %s, type: %s check no multi stream users failed for reinplace. "
                     "The users of the mutated input node have multiple streams. All the users are %s.", check_name,
                     node.name, node.target, users)
            return False