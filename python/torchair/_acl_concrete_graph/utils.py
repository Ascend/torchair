from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Any, Dict, Tuple, Union
import weakref

import torch
from torch.types import Device, Number


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
    captured_inputs: List[torch.Tensor]
    outputs_meta: List[Union[TensorMetadata, Number]]
    outputs_weakref: List[WeakRef]
    mem_state_after_capture: Any
    is_first_replay: bool
    retained_outputs: List[torch.Tensor] = None


def get_tensor_metadata(x):
    """
    Just record meta data for a pytorch tensor object.
    """

    if isinstance(x, torch.Tensor):
        return TensorMetadata(
            size=x.shape,
            stride=x.stride(),
            nbytes=x.untyped_storage().nbytes(),
            dtype=x.dtype,
            data_ptr=x.untyped_storage().data_ptr(),
            storage_offset=x.storage_offset(),
            device=x.device,
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
