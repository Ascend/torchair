import json
import os
from npu_extension_for_inductor.common.op_code import OpCode, OpProto


def codegen_stub(*args, **kwargs):
    # 在这里粘贴@liqiduan准备的stub代码
    # 在这里粘贴@liqiduan准备的stub代码
    # 在这里粘贴@liqiduan准备的stub代码

    return None


def codegen(graph):
    if os.environ.get('ASCIR_NOT_READY', None) == "1":
        return codegen_stub(graph)

    from pyautofuse import Autofuser, ascir
    print("Graph before fuse")
    print(ascir.utils.debug_str(graph))

    fuser = Autofuser({})
    fused_graph = fuser.autofuse(graph)
    print("Graph after fuse")
    for i in range(len(fused_graph)):
        print("Impl ", i)
        print(ascir.utils.debug_str(fused_graph[i]))

    op_proto, tiling_def, host_tiling, op_kernel = fuser.codegen(graph, fused_graph)
    return OpCode(OpProto(op_proto), tiling_def, host_tiling, op_kernel)