import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import traceable_collective_remaps, all_gather_tensor
from torch.distributed.distributed_c10d import (
    _all_gather_base as legacy_all_gather_base,
    all_gather_into_tensor as legacy_allgather,
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
    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"

    group = group or dist.group.WORLD
    assert group is not None

    return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))


def adjust_traceable_collective_remaps():
    if torch.__version__ >= '2.3.1':
        traceable_collective_remaps.update({
            legacy_all_gather_base: all_gather_tensor_inplace_fixed,
            legacy_allgather: all_gather_tensor_inplace_fixed,
        })
        setattr(torch.distributed._functional_collectives,
                'all_gather_tensor_inplace_fixed', all_gather_tensor_inplace_fixed)
