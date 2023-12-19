import json
import os
from npu_extension_for_inductor.common.op_code import OpCode, OpProto


def codegen_stub(*args, **kwargs):
    # 在这里粘贴@liqiduan准备的stub代码
    # 在这里粘贴@liqiduan准备的stub代码
    # 在这里粘贴@liqiduan准备的stub代码

    return OpCode(test_op_proto, test_op_tiling, test_op_host_code, test_op_device_code)


def codegen(graph):
    if os.environ.get('ASCIR_NOT_READY', None):
        return codegen_stub(graph)

    from pyautofuse import Autofuser
    fuser = Autofuser({})
    fused_graph = fuser.autofuse(graph)
    op_proto, tiling_def, host_tiling, op_kernel = fuser.codegen(fused_graph)
    return OpCode(op_proto, tiling_def, host_tiling, op_kernel)
