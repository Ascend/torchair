__all__ = []

import torch

from torchair.core.utils import logger
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather import npu_all_gather_patch_dist, \
npu_allgather_in_tensor_patch_dist


logger.warning_once(f'The usage of torchair.ge_concrete_graph .* will not be supported in the future,'
                    f' please complete the API switch as soon as possible.')

torch.distributed.all_gather = npu_all_gather_patch_dist
torch.distributed.all_gather_into_tensor = npu_allgather_in_tensor_patch_dist

