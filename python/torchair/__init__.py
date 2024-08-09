import os
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
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allreduce import npu_allreduce_patch_dist, \
patch_for_deepspeed_allreduce
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_allgather import npu_all_gather_patch_dist, \
npu_allgather_in_tensor_patch_dist
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_broadcast import npu_broadcast_patch_dist
from torchair._ge_concrete_graph.ge_converter.experimental.hcom_alltoall import npu_all_to_all_single_patch_dist, \
npu_all_to_all_patch_dist
import torchair.inference
import torchair.llm_datadist
import torchair.ops
import torchair.ge

__all__ = ['get_compiler', 'get_npu_backend', 'dynamo_export', 'CompilerConfig',
           'use_internal_format_weight', 'logger', 'register_fx_node_ge_converter',
           'patch_for_hcom']

# Dependency library version verification
protobuf_version = pkg_resources.get_distribution("protobuf").version

if not pkg_resources.parse_version(protobuf_version) >= pkg_resources.parse_version("3.13"):
    raise AssertionError


def register_fx_node_ge_converter(aten_op):
    return _register_fx_node_ge_converter(aten_op)


def patch_for_hcom():
    patch_for_deepspeed_allreduce()
    torch.distributed.all_reduce = npu_allreduce_patch_dist
    torch.distributed.all_gather = npu_all_gather_patch_dist
    torch.distributed.all_gather_into_tensor = npu_allgather_in_tensor_patch_dist
    torch.distributed.broadcast = npu_broadcast_patch_dist
    torch.distributed.all_to_all_single = npu_all_to_all_single_patch_dist
    torch.distributed.all_to_all = npu_all_to_all_patch_dist
