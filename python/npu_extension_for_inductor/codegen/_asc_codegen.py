import os
import sys
import itertools
from types import ModuleType
from typing import Union

import torch
from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.asc_graph import ASCGraph, FusedASCGraph
from npu_extension_for_inductor.common.debug import save_asserts
from npu_extension_for_inductor.common.utils import camel_to_snake
from npu_extension_for_inductor.common.utils import load_autofuser


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


def codegen_kernel_def(graph: FusedASCGraph, var_name=None) -> str:
    var_name = var_name or graph.name
    kernel_def = IndentedBuffer()
    graph_fn = graph.name

    graph_py_code = IndentedBuffer()
    graph_py_code.splice(f"from pyautofuse import ascir")
    graph_py_code.splice(f'from pyautofuse import Autofuser, AutofuserOptions')
    graph_py_code.splice(graph.codegen())
    graph_py_code.splice(f'''
    fuser = Autofuser(AutofuserOptions(graph_type=1))
    scheduled_{graph.name} = fuser.schedule({graph.name})
    tiling_def, host_impl, device_impl = fuser.codegen(scheduled_{graph.name})
    ''')
    save_asserts(graph.name, graph_py_code.getvalue(), 'asc_graph.py')

    local_vars = dict()
    with load_autofuser(graph.name):
        try:
            exec(compile(graph_py_code.getvalue(), '<string>', 'exec'), globals(), local_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to execute graph code:{graph_py_code.getvalue()} {e}") from e

    artifacts = dict()
    artifacts['name'] = graph.name
    artifacts['tiling_def'] = local_vars.get('tiling_def')
    artifacts['host_impl'] = local_vars.get('host_impl')
    artifacts['device_impl'] = local_vars.get('device_impl')
    artifacts['cpp_wrapper'] = codegen_cpp_wrapper(graph)

    if not all(v.strip() for v in artifacts.values()):
        raise RuntimeError(f"Failed to generate artifacts for kernel {graph.name}: {artifacts}")

    kernel_def.writeline(f"{graph_fn}_artifacts = {{}}")
    for k, v in artifacts.items():
        kernel_def.splice(f"{graph_fn}_artifacts['{k}'] = '''{v}'''")
    kernel_def.writeline(
        f"{var_name} = async_compile_ascendc(globals().get('async_compile', None), {graph_fn}_artifacts)")

    return kernel_def.getvalue()


def codegen_cpp_wrapper(graph: FusedASCGraph):
    wrapper = IndentedBuffer()
    inputs = [TensorArg(v) for v in graph.inputs]
    outputs = [TensorArg(v) for v in graph.outputs]
    symbols = [SymArg(str(v)) for v in graph.size_vars]
    stream = StreamArg("stream")
    tiling_dtype = f"AutofuseTilingData"

    all_args = [TensorArg(v) for v in graph.args] + symbols + [stream]
    signature = ', '.join([v.signature for v in all_args])
    buffer_assign = ''
    for in_name, out_name in zip(graph.inputs + graph.outputs, graph.inputs_outer + graph.outputs_outer):
        buffer_assign += f'\n    auto *{in_name} = {out_name};'
        buffer_assign += f'\n    DLOG() << "{in_name}: " << {in_name} << std::endl;'

    tiling_args = [v.name for v in symbols]
    tiling_signature = [v.signature for v in symbols]
    tiling_signature.append(f"{tiling_dtype} *tiling_data")
    tiling_signature.append(f"int64_t *workspace_size")
    tiling_signature.append(f"int64_t *block_dim")

    workspaces = [TensorArg("workspace")]
    launch_args = [v.name for v in itertools.chain(inputs, outputs, workspaces)]
    launch_signature = ["int64_t block_dim", "void *stream"]
    launch_signature.extend([v.signature for v in itertools.chain(inputs, outputs, workspaces)])
    launch_signature.append(f"{tiling_dtype} *tiling_data")

    wrapper.splice(f'''
typedef int64_t (*TilingFuncType)({', '.join(tiling_signature)});
typedef int64_t (*LaunchFuncType)({', '.join(launch_signature)});
static TilingFuncType tiling_fn = reinterpret_cast<TilingFuncType>(GetFunc("AutofuseTiling"));
static LaunchFuncType launch_fn = reinterpret_cast<LaunchFuncType>(GetFunc("AutofuseLaunch"));

extern "C" int wrapper({signature}) {{
    {tiling_dtype} tiling_data;
    int64_t workspace_size = 0;
    int64_t block_dim = 0;
    if (tiling_fn == nullptr || launch_fn == nullptr) {{
        if (tiling_fn == nullptr) std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
        if (launch_fn == nullptr) std::cerr << "{graph.name} kernel launch func not found" << std::endl;
        return -1;
    }}
    int64_t result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim"])});
    if (result != 0) {{
        return -1;
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        return -1;
    }}

    DLOG() << "block_dim: " << block_dim << std::endl;
    DLOG() << "stream: " << current_stream << std::endl;
    DLOG() << "workspace_size: " << workspace_size << std::endl;

    void *workspace = nullptr;
    workspace_size = workspace_size < 0 ? 0 : workspace_size;
    if (workspace_size > 0) {{
        workspace = MallocWorkspace(workspace_size, current_stream);
        if (workspace == nullptr) {{
            return -1;
        }}
    }}
    DLOG() << "workspace: " << workspace << std::endl;

    {buffer_assign}

    result = launch_fn({', '.join(["block_dim", "current_stream"] + launch_args + ["&tiling_data"])});
    if (workspace != nullptr) {{
        FreeWorkspace(workspace);
    }}
    if (result != 0) {{
        return -1;
    }}
    return 0;
}}
    ''')

    return wrapper.getvalue()
