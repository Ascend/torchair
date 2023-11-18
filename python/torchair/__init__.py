import torch
import pkg_resources

from torchair.npu_fx_compiler import get_compiler
from torchair.npu_fx_compiler import get_npu_backend
from torchair.dynamo_export import dynamo_export
from torchair.core.backend import stupid_repeat
from torchair.configs.compiler_config import CompilerConfig
from torchair.ge_concrete_graph import ge_converter


# Dependency library version verification
protobuf_version = pkg_resources.get_distribution("protobuf").version

assert pkg_resources.parse_version(protobuf_version) >= pkg_resources.parse_version("3.13")
