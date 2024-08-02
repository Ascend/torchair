import os
import json
import itertools
from types import ModuleType

from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.asc_graph import ASCGraph
from npu_extension_for_inductor.common.debug import save_asserts


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
    if os.getenv("NPU_INDUCTOR_DUMMY_KERNEL", None) == "1":
        kernel_def.writeline(
            "from npu_extension_for_inductor.compiler._aclnn_compiler import DummyNpuInductorKernel")
        kernel_def.writeline(f"{var_name} = DummyNpuInductorKernel('{graph.name}')")
    else:
        graph_py_code = IndentedBuffer()
        if os.getenv('ASCIR_NOT_READY', None) == "1":
            graph_py_code.splice("from npu_extension_for_inductor.common.revert_ascir import RevertAscir")
            graph_py_code.splice("ascir = RevertAscir()")
            graph_py_code.splice(graph.codegen())
            graph_py_code.splice(f'tiling_def, host_impl, device_impl = {graph.name}.codegen()')
        else:
            graph_py_code.splice(f"from pyautofuse import ascir")
            graph_py_code.splice(f'from pyautofuse import Autofuser, AutofuserOptions')
            graph_py_code.splice(graph.codegen())
            graph_py_code.splice(f'''
            fuser = Autofuser(AutofuserOptions())
            fused_{graph.name} = fuser.autofuse({graph.name})
            op_proto, tiling_def, host_impl, device_impl = fuser.codegen({graph.name}, fused_{graph.name})
            ''')
        save_asserts(graph.name, graph_py_code.getvalue(), 'asc_graph_python.py')

        codegen_mod = ModuleType('codegen_mod')
        local_vars = dict()
        exec(compile(graph_py_code.getvalue(), '<string>', 'exec'), codegen_mod.__dict__, local_vars)

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
    all_args = inputs + outputs + workspaces + symbols + [stream]

    signature = ', '.join([v.signature for v in all_args])
    tiling_args = [v.name for v in symbols]
    kernel_args = [v.name for v in itertools.chain(inputs, outputs, workspaces)]
    wrapper.splice(f'''
    extern "C" int wrapper({signature}) {{
        {graph.name}TilingData tiling_data;
        int64_t workspace_size = 0;
        int64_t block_dim = 0;
        void *current_stream = (stream == nullptr) ? c10_npu::getCurrentNPUStream().stream() : stream;
        if (tiling_fn == nullptr || kernel_fn == nullptr) {{
            if (tiling_fn == nullptr) std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
            if (kernel_fn == nullptr) std::cerr << "{graph.name} kernel launch func not found" << std::endl;
            return -1;
        }}
        tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim"])});
        kernel_fn({', '.join(["block_dim", "current_stream"] + kernel_args + ["&tiling_data"])});
        return 0;
    }}
    ''')

    return wrapper.getvalue()
