from .npu import NPUScheduling, NpuWrapperCodeGen
from torch._inductor.codegen.common import register_backend_for_device

register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)
