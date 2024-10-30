import os
import sys
import itertools
from types import ModuleType

import torch
from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.debug import save_asserts
from npu_extension_for_inductor.common.utils import camel_to_snake


class KernelArg:
    def __init__(self, name, c_type):
        self.name = name
        self.c_type = c_type

    def __repr__(self):
        return self.name

    @property
    def signature(self):
        if self.c_type.endswith("*"):
            return f"{self.c_type}{self.name}"
        return f"{self.c_type} {self.name}"


class TensorArg(KernelArg):
    def __init__(self, name):
        super().__init__(name, "void *")


class StreamArg(KernelArg):
    def __init__(self, name):
        super().__init__(name, "void *")


class SymArg(KernelArg):
    def __init__(self, name):
        super().__init__(name, "int64_t")


def codegen_kernel_def(graph: ASCGraph, var_name=None) -> str:
    var_name = var_name or graph.name
    kernel_def = IndentedBuffer()
    graph_fn = graph.name

    graph_py_code = IndentedBuffer()
    graph_py_code.splice(f"from pyautofuse import ascir")
    graph_py_code.splice(f'from pyautofuse import Autofuser, AutofuserOptions')
    graph_py_code.splice(graph.codegen())
    graph_py_code.splice(f'''
    fuser = Autofuser(AutofuserOptions())
    fused_{graph.name} = fuser.autofuse({graph.name})
    op_proto, tiling_def, host_impl, device_impl = fuser.codegen({graph.name}, fused_{graph.name})
    ''')
    save_asserts(graph.name, graph_py_code.getvalue(), 'asc_graph_python.py')

    local_vars = dict()
    exec(compile(graph_py_code.getvalue(), '<string>', 'exec'), globals(), local_vars)

    artifacts = dict()
    artifacts['name'] = graph.name
    artifacts['tiling_def'] = local_vars.get('tiling_def')
    artifacts['host_impl'] = local_vars.get('host_impl')
    artifacts['device_impl'] = local_vars.get('device_impl')
    artifacts['cpp_wrapper'] = codegen_cpp_wrapper(graph)

    kernel_def.writeline(f"{graph_fn}_artifacts = {{}}")
    for k, v in artifacts.items():
        kernel_def.splice(f"{graph_fn}_artifacts['{k}'] = '''{v}'''")
    kernel_def.writeline(f"{var_name} = npu_compiler.aclnn({graph_fn}_artifacts)")

    return kernel_def.getvalue()


def codegen_cpp_wrapper(graph: ASCGraph):
    wrapper = IndentedBuffer()
    inputs = [TensorArg(v) for v in graph.inputs]
    outputs = [TensorArg(v) for v in graph.outputs]
    workspaces = [TensorArg("workspace")]
    symbols = [SymArg(str(v)) for v in sorted(list(graph.size_vars))]
    stream = StreamArg("stream")
    tiling_dtype = f"{graph.name}TilingData"

    unique_outer = sorted(set(graph.inputs_outer + graph.outputs_outer))
    all_args = [TensorArg(v) for v in unique_outer] + workspaces + symbols + [stream]
    signature = ', '.join([v.signature for v in all_args])
    buffer_assign = ''
    for in_name, out_name in zip(graph.inputs + graph.outputs, graph.inputs_outer + graph.outputs_outer):
        buffer_assign += f'\nauto *{in_name} = {out_name};'
        buffer_assign += f'\nDLOG() << "{in_name}: " << {in_name} << std::endl;'

    tiling_args = [v.name for v in symbols]
    launch_args = [v.name for v in itertools.chain(inputs, outputs, workspaces)]

    tiling_signature = [v.signature for v in symbols]
    tiling_signature.append(f"{tiling_dtype} *tiling_data")
    tiling_signature.append(f"int64_t *workspace_size")
    tiling_signature.append(f"int64_t *block_dim")

    launch_signature = ["int64_t block_dim", "void *stream"]
    launch_signature.extend([v.signature for v in itertools.chain(inputs, outputs, workspaces)])
    launch_signature.append(f"{tiling_dtype} *tiling_data")

    wrapper.splice(f'''
    typedef int (*TilingFuncType)({', '.join(tiling_signature)});
    typedef int (*LaunchFuncType)({', '.join(launch_signature)});
    static TilingFuncType tiling_fn = reinterpret_cast<TilingFuncType>(GetFunc("{graph.name}TilingFunc"));
    static LaunchFuncType launch_fn = reinterpret_cast<LaunchFuncType>(GetFunc("aclrtlaunch_{camel_to_snake(graph.name)}"));
    extern "C" int wrapper({signature}) {{
        {tiling_dtype} tiling_data;
        int64_t workspace_size = 0;
        int64_t block_dim = 0;
        if (tiling_fn == nullptr || launch_fn == nullptr) {{
            if (tiling_fn == nullptr) std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
            if (launch_fn == nullptr) std::cerr << "{graph.name} kernel launch func not found" << std::endl;
            return -1;
        }}
        int result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim"])});
        if (result != 0) {{
            return result;
        }}
        {buffer_assign}
        DLOG() << "block_dim: " << block_dim << std::endl;
        DLOG() << "stream: " << GetStream(stream) << std::endl;
        return launch_fn({', '.join(["block_dim", "GetStream(stream)"] + launch_args + ["&tiling_data"])});
    }}
    ''')

    return wrapper.getvalue()
