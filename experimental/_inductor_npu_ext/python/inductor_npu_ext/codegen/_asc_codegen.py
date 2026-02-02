import itertools

from typing import Dict
from torch._inductor.codegen.common import IndentedBuffer
from inductor_npu_ext.common.asc_graph import FusedASCGraph
from inductor_npu_ext.common.debug import save_asserts
from inductor_npu_ext.common.utils import load_autofuser


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


def codegen_kernel_def(graph: FusedASCGraph) -> Dict[str, str]:
    graph_py_code = IndentedBuffer()
    graph_py_code.splice(f"from autofuse.pyautofuse import ascir")
    graph_py_code.splice(f'from autofuse.pyautofuse import Autofuser, AutofuserOptions')
    graph_py_code.splice(f'''
def Mod(x, y):
    return x % y
def PythonMod(x, y):
    return x % y
''')
    graph_py_code.splice(graph.codegen(graph.name))
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
    return artifacts


def codegen_cpp_wrapper(graph: FusedASCGraph):
    wrapper = IndentedBuffer()
    inputs = [TensorArg(f"asc_input{v}") for v in range(len(graph.inputs))]
    outputs = [TensorArg(f"asc_output{v}") for v in range(len(graph.outputs))]
    symbols = [SymArg(str(v)) for v in graph.size_vars]
    stream = StreamArg("stream")
    tiling_dtype = f"AutofuseTilingData"

    all_args = [TensorArg(v) for v in graph.args] + symbols + [stream]
    signature = ', '.join([v.signature for v in all_args])
    buffer_assign = ''
    for in_arg, out_name in zip(inputs + outputs, graph.inputs_outer + graph.outputs_outer):
        buffer_assign += f'\n    auto *{in_arg.name} = {out_name};'
        buffer_assign += f'\n    DLOG() << "{in_arg.name}: " << {in_arg.name} << std::endl;'

    tiling_args = [v.name for v in symbols]
    tiling_signature = [v.signature for v in symbols]
    tiling_signature.append(f"{tiling_dtype} *tiling_data")
    tiling_signature.append(f"uint32_t *workspace_size")
    tiling_signature.append(f"uint32_t *block_dim")
    tiling_signature.append(f"void *resource_limit")

    workspaces = [TensorArg("workspace")]
    launch_args = [v.name for v in itertools.chain(inputs, outputs, workspaces)]
    launch_signature = ["uint32_t block_dim", "void *stream"]
    launch_signature.extend([v.signature for v in itertools.chain(inputs, outputs, workspaces)])
    launch_signature.append(f"{tiling_dtype} *tiling_data")

    wrapper.splice(f'''
namespace {{
    typedef int64_t (*TilingFuncType)({', '.join(tiling_signature)});
    typedef int64_t (*LaunchFuncType)({', '.join(launch_signature)});
    TilingFuncType tiling_fn = nullptr;
    LaunchFuncType launch_fn = nullptr;
    void *handle = nullptr;

    const bool static_tiling = {'true' if len(tiling_args) == 0 else 'false'};
    uint32_t default_workspace_size = 0;
    uint32_t default_block_dim = 0;
    {tiling_dtype} default_tiling_data = {{}};

    std::mutex mtx;
    bool initialized = false;
}}  // namespace
__attribute__((destructor)) static void DeInit() {{
    if (handle != nullptr) {{
        dlclose(handle);
        handle = nullptr;
    }}
}}
''')

    if len(tiling_args) == 0:
        wrapper.splice(f'''
        static int InitDefaultTiling() {{
            if (tiling_fn == nullptr) {{
                std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
                return -1;
            }}
            const int64_t result = tiling_fn({', '.join(['&default_tiling_data', '&default_workspace_size', '&default_block_dim', 'nullptr'])});
            if (result != 0) {{
                std::cerr << "{graph.name} kernel tiling failed" << std::endl;
                return -1;
            }}
            return 0;
        }}
        ''')
    else:
        wrapper.splice(f'''
        static int InitDefaultTiling() {{
            return 0;
        }}
        ''')

    wrapper.splice(f'''
extern "C" int init(const char *kernel_file) {{
    std::lock_guard<std::mutex> lock(mtx);
    if (initialized) {{
        return 0;
    }}
    handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {{
        std::cerr << "Kernel load failed for {graph.name}" << std::endl;
        return -1;
    }}
    tiling_fn = reinterpret_cast<TilingFuncType>(dlsym(handle, "AutofuseTiling"));
    if (tiling_fn == nullptr) {{
        std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
        dlclose(handle);
        handle = nullptr;
        return -1;
    }}
    launch_fn = reinterpret_cast<LaunchFuncType>(dlsym(handle, "AutofuseLaunch"));
    if (launch_fn == nullptr) {{
        std::cerr << "{graph.name} kernel launch func not found" << std::endl;
        dlclose(handle);
        handle = nullptr;
        return -1;
    }}
    if (InitDefaultTiling() != 0) {{
        dlclose(handle);
        handle = nullptr;
        return -1;
    }}
    initialized = true;
    return 0;
}}

extern "C" int wrapper({signature}) {{
    uint32_t workspace_size = default_workspace_size;
    uint32_t block_dim = default_block_dim;
    int64_t result = 0;
    if (tiling_fn == nullptr || launch_fn == nullptr) {{
        if (tiling_fn == nullptr) std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
        if (launch_fn == nullptr) std::cerr << "{graph.name} kernel launch func not found" << std::endl;
        return -1;
    }}

    {tiling_dtype} tiling_data = default_tiling_data;
    if (!static_tiling) {{
        result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"])});
        if (result != 0) {{
            std::cerr << "{graph.name} kernel tiling failed" << std::endl;
            return -1;
        }}
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        std::cerr << "{graph.name} kernel get stream failed" << std::endl;
        return -1;
    }}

    DLOG() << "Launch args for {graph.name}:" << std::endl;
    DLOG() << "block_dim: " << block_dim << std::endl;
    DLOG() << "stream: " << current_stream << std::endl;
    DLOG() << "workspace_size: " << workspace_size << std::endl;

    void *workspace = nullptr;
    if (workspace_size > 0) {{
        workspace = MallocWorkspace(workspace_size, current_stream);
        if (workspace == nullptr) {{
            std::cerr << "{graph.name} kernel malloc workspace failed" << std::endl;
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
        std::cerr << "{graph.name} kernel launch failed" << std::endl;
        return -1;
    }}
    return 0;
}}
    ''')

    return wrapper.getvalue()
