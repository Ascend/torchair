from typing import Dict, Callable


def aclnn(artifacts: Dict) -> Callable:
    from . import _aclnn_compiler
    return _aclnn_compiler.compile_ascendc(artifacts)
