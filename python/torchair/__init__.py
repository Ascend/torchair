import os
import torch
import pkg_resources

from torchair.npu_fx_compiler import get_compiler
from torchair.npu_fx_compiler import get_npu_backend
from torchair.dynamo_export import dynamo_export
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph import ge_converter
from torchair.experimental.inference import use_internal_format_weight

if os.getenv("ENABLE_ACLNN") is None:
    os.environ["ENABLE_ACLNN"] = "true"

__all__ = ['get_compiler', 'get_npu_backend', 'dynamo_export', 'CompilerConfig', 'use_internal_format_weight']

# Dependency library version verification
protobuf_version = pkg_resources.get_distribution("protobuf").version

assert pkg_resources.parse_version(protobuf_version) >= pkg_resources.parse_version("3.13")
