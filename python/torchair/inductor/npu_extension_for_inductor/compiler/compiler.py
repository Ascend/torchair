from typing import Callable

from npu_extension_for_inductor.common.op_code import OpCode


def aclnn(src: OpCode) -> Callable:
    from . import aclnn_compiler
    return aclnn_compiler.compile(src)