__all__ = ['get_compiler', 'get_npu_backend', 'dynamo_export', 'CompilerConfig',
           'use_internal_format_weight', 'logger', 'register_fx_node_ge_converter',
           'patch_for_hcom']

import os
import sys
import torch
import pkg_resources

from torchair.npu_fx_compiler import get_compiler
from torchair.npu_fx_compiler import get_npu_backend
from torchair.npu_export import dynamo_export
from torchair.configs.compiler_config import CompilerConfig
from torchair._ge_concrete_graph import ge_converter
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter as _register_fx_node_ge_converter
from torchair.experimental.inference import use_internal_format_weight
from torchair.core.utils import logger

import torchair.inference
import torchair.llm_datadist
import torchair.ops
import torchair.ge
import torchair.scope


# Dependency library version verification
protobuf_version = pkg_resources.get_distribution("protobuf").version

if not pkg_resources.parse_version(protobuf_version) >= pkg_resources.parse_version("3.13"):
    raise AssertionError("protobuf_version must satisfied >=3.13")

# before patch, backup function call for torch_npu.distributed.xxx
try:
    import torch_npu
    ALL_GATHER_INTO_TENSOR_UNEVEN = torch_npu.distributed.all_gather_into_tensor_uneven
    REDUCE_SCATTER_TENSOR_UNEVEN = torch_npu.distributed.reduce_scatter_tensor_uneven
except (ImportError, AttributeError) as e:
    ALL_GATHER_INTO_TENSOR_UNEVEN = None
    REDUCE_SCATTER_TENSOR_UNEVEN = None


def register_fx_node_ge_converter(aten_op):
    return _register_fx_node_ge_converter(aten_op)


def patch_for_hcom():
    if torch.__version__ >= "2.3.1":
        logger.warning(f'For versions 2.3.1 and above of PyTorch, there is no need to call patch_for_hcom anymore.')

    from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allreduce import npu_allreduce_patch_dist, \
        patch_for_deepspeed_allreduce
    from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather import (npu_all_gather_patch_dist,
        npu_allgather_in_tensor_patch_dist, npu_allgather_into_tensor_uneven_patch_dist)
    from torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast import npu_broadcast_patch_dist
    from torchair._ge_concrete_graph.ge_converter.experimental.hcom_alltoall import (npu_all_to_all_single_patch_dist,
                                                                                     npu_all_to_all_patch_dist)
    patch_for_deepspeed_allreduce()
    torch.distributed.all_reduce = npu_allreduce_patch_dist
    torch.distributed.all_gather = npu_all_gather_patch_dist
    torch.distributed.all_gather_into_tensor = npu_allgather_in_tensor_patch_dist
    torch.distributed.broadcast = npu_broadcast_patch_dist
    torch.distributed.all_to_all_single = npu_all_to_all_single_patch_dist
    torch.distributed.all_to_all = npu_all_to_all_patch_dist

    if torch.__version__ < "2.3.1":
        if 'torch_npu' not in sys.modules:
            logger.warning(f'The patch for torch_npu.distributed.xxx will only be enabled in a torch npu env.'
                        'When there is no torch_npu in the env, skip patch for torch_npu.')
            return
        from torchair._ge_concrete_graph.ge_converter.experimental.hcom_reducescatter import \
            npu_reduce_scatter_tensor_uneven_patch_dist
        torch_npu.distributed.all_gather_into_tensor_uneven = npu_allgather_into_tensor_uneven_patch_dist
        torch_npu.distributed.reduce_scatter_tensor_uneven = npu_reduce_scatter_tensor_uneven_patch_dist

