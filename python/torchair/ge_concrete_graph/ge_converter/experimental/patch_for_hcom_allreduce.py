__all__ = []

import torch

from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allreduce import npu_allreduce_patch_dist, \
patch_for_deepspeed_allreduce


torch.distributed.all_reduce = npu_allreduce_patch_dist
patch_for_deepspeed_allreduce()
