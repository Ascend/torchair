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


def get_inplace_op_mutated_input_users(original_node: Node,
                                       mutated_input_indices: List[int]) -> List[List[Node]]:
    """
    Returns all downstream users of the mutated input nodes from an in-place operation.

    For each specified input index that is mutated by the in-place operation,
    collects all subsequent nodes in the graph that consume the mutated value
    (including indirect users through intermediate nodes).

    Args:
        original_node: The original functionalized node that will be reinplaced.
        mutated_input_indices: List of indices pointing to which inputs will be mutated.

    Returns:
        List of user node lists.
    """

    all_inputs = [original_node.args[idx] for idx in mutated_input_indices]

    all_inputs_users = []
    for input_node in all_inputs:
        cur_input_users = []
        to_be_replaced_copy_nodes = None
        # part 1: all users of input node
        for user_node in input_node.users:
            if user_node.op == "call_function" and user_node.target == torch.ops.aten.copy_.default:
                # user copy_ will be replace by input
                to_be_replaced_copy_nodes = user_node
                continue

            cur_input_users.append(user_node)

        if to_be_replaced_copy_nodes is None:
            logger.debug("All the users of placeholder node[%s] are nodes: %s.",
                         input_node.name, cur_input_users)
            all_inputs_users.append(cur_input_users)
            continue

        # part 2: all users of functionalized op output
        # Only when multi reinplace is need,
        # because functionalized op output(get_item) will be replace by input only in multi reinplace case.
        candidates = []
        if to_be_replaced_copy_nodes.args[1].target == operator.getitem:
            candidates = to_be_replaced_copy_nodes.args[1].users

        for user_node in candidates:
            if user_node == to_be_replaced_copy_nodes:
                # user copy_ will be replace by input
                continue
            cur_input_users.append(user_node)

        # part 3: all users of copy_
        for user_node in to_be_replaced_copy_nodes.users:
            cur_input_users.append(user_node)

        logger.debug("All the users of placeholder node[%s] are nodes: %s.",
                     input_node.name, cur_input_users)
        all_inputs_users.append(cur_input_users)

    return all_inputs_users
