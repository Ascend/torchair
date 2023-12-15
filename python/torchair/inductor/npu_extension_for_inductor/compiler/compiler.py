from types import ModuleType
from npu_extension_for_inductor.common.op_code import OpCode


def aclnn(src: OpCode) -> ModuleType:
    from . import aclnn_compiler
    return aclnn_compiler.compile(src)
