import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import traceable_collective_remaps, all_gather_tensor
from torch.distributed.distributed_c10d import (
    _all_gather_base as legacy_all_gather_base,
    all_gather_into_tensor as legacy_allgather,
    all_to_all as legacy_all_to_all,
    broadcast as legacy_broadcast,
    send as legacy_send,
    recv as legacy_recv,
)


# This function is used to fix the issues mentioned in PR: https://github.com/pytorch/pytorch/pull/143700
#  on lower versions of PyTorch.
def all_gather_tensor_inplace_fixed(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group=None,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0,
):
    if async_op:
        raise ValueError("Can't remap async version of inplace op to functional collective")

    group = group or dist.group.WORLD
    if group is None:
        raise ValueError("Group is not set")

    # fix stack case mentioned in issue: https://github.com/pytorch/pytorch/issues/155632
    result = all_gather_tensor(input_tensor, gather_dim, group, tag)
    if result.shape == output_tensor.shape:
        return output_tensor.copy_(result)

    stacked_result = torch.stack(torch.split(result, input_tensor.shape[0], dim=0), dim=0)
    if stacked_result.shape == output_tensor.shape:
        return output_tensor.copy_(stacked_result)

    msg = f"Input shape {input_tensor.shape} and output shape {output_tensor.shape} are not compatible for all_gather_into_tensor. Input must be stacked or concatenated to create output."
    raise ValueError(msg)


def adjust_traceable_collective_remaps():
    if torch.__version__ >= '2.3.1':
        from torchair._ge_concrete_graph.ge_converter.experimental.hcom_alltoall import npu_all_to_all_patch_dist
        from torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast import npu_broadcast_patch_dist
        from torchair._ge_concrete_graph.ge_converter.experimental.hcom_send_recv import (
            npu_send_patch_dist,
            npu_recv_patch_dist
        )
        traceable_collective_remaps.update({
            legacy_all_gather_base: all_gather_tensor_inplace_fixed,
            legacy_allgather: all_gather_tensor_inplace_fixed,
            legacy_all_to_all: npu_all_to_all_patch_dist,
            legacy_broadcast: npu_broadcast_patch_dist,
            legacy_send: npu_send_patch_dist,
            legacy_recv: npu_recv_patch_dist,
        })
        setattr(torch.distributed._functional_collectives,
                'all_gather_tensor_inplace_fixed', all_gather_tensor_inplace_fixed)
        setattr(torch.distributed._functional_collectives,
                'npu_all_to_all_patch_dist', npu_all_to_all_patch_dist)
        setattr(torch.distributed._functional_collectives,
                'npu_broadcast_patch_dist', npu_broadcast_patch_dist)
        setattr(torch.distributed._functional_collectives,
                'npu_send_patch_dist', npu_send_patch_dist)
        setattr(torch.distributed._functional_collectives,
                'npu_recv_patch_dist', npu_recv_patch_dist)
        try:
            from torch_npu.distributed import all_gather_into_tensor_uneven, reduce_scatter_tensor_uneven
            from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather import (
                npu_allgather_into_tensor_uneven_patch_dist)
            from torchair._ge_concrete_graph.ge_converter.experimental.hcom_reducescatter import (
                npu_reduce_scatter_tensor_uneven_patch_dist)
            traceable_collective_remaps.update({
                all_gather_into_tensor_uneven: npu_allgather_into_tensor_uneven_patch_dist,
                reduce_scatter_tensor_uneven: npu_reduce_scatter_tensor_uneven_patch_dist,
            })
            setattr(torch.distributed._functional_collectives,
                    'npu_allgather_into_tensor_uneven_patch_dist', npu_allgather_into_tensor_uneven_patch_dist)
            setattr(torch.distributed._functional_collectives,
                    'npu_reduce_scatter_tensor_uneven_patch_dist', npu_reduce_scatter_tensor_uneven_patch_dist)
        except ImportError:
            return
