__all__ = ["create_npu_tensors"]

from typing import List
import torch
from ._tensor_utils import _create_npu_tensors


def create_npu_tensors(shape: List[int], dtype: torch.dtype, addresses: List[int]) -> List[torch.Tensor]:
    return _create_npu_tensors(shape, dtype, addresses)
