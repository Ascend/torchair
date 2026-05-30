import itertools

from typing import Dict
from torch._inductor.codegen.common import IndentedBuffer
from ..common.asc_graph import FusedASCGraph
from ..common.debug import save_asserts
from ..common.utils import load_autofuser
from .. import config as ext_config


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
    stream = StreamArg("stream")
    tiling_dtype = f"AutofuseTilingData"

    from torch._inductor.codegen.common import TensorArg as InductorTensorArg
    from torch._inductor.codegen.common import SizeArg as InductorSizeArg

    tensor_args = []
    symbol_args = []
    for v in graph.args:
        if isinstance(v, InductorTensorArg):
            tensor_args.append(TensorArg(v.name))
        elif isinstance(v, InductorSizeArg):
            symbol_args.append(SymArg(v.name))
        else:
            raise NotImplementedError(f"Unsupported arg type: {type(v)}")

    all_args = tensor_args + symbol_args + [stream]
    signature = ', '.join([v.signature for v in all_args])
    buffer_assign = ''
    for in_arg, out_name in zip(inputs + outputs, graph.inputs_outer + graph.outputs_outer):
        buffer_assign += f'\n    auto *{in_arg.name} = {out_name};'
        buffer_assign += f'\n    DLOG() << "{in_arg.name}: " << {in_arg.name} << std::endl;'
    # One more indent when buffer_assign is emitted inside the TaskQueue lambda
    if ext_config._enable_taskqueue_mode != 2:
        buffer_assign_in_taskqueue1 = buffer_assign.replace('\n    ', '\n    ')
    else:
        buffer_assign_in_taskqueue2 = buffer_assign.replace('\n    ', '\n        ')

    tiling_args = [v.name for v in symbol_args]
    tiling_signature = [v.signature for v in symbol_args]
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
''')

    # taskqueue=0:同步执行
    # taskqueue=1:静态tiling+动态tiling+workspace 在queue外执行，kernel在queue内执行
    if ext_config._enable_taskqueue_mode != 2:
        wrapper.splice(f'''
extern "C" int wrapper({signature}) {{
    if (tiling_fn == nullptr || launch_fn == nullptr) {{
        if (tiling_fn == nullptr) std::cerr << "{graph.name} tiling func null" << std::endl;
        if (launch_fn == nullptr) std::cerr << "{graph.name} launch func null" << std::endl;
        return -1;
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        std::cerr << "{graph.name} GetStream failed in wrapper thread" << std::endl;
        return -1;
    }}

    uint32_t workspace_size = default_workspace_size;
    uint32_t block_dim = default_block_dim;
    {tiling_dtype} tiling_data = default_tiling_data;
    if (!static_tiling) {{
        int64_t tiling_result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"])});
        if (tiling_result != 0) {{
            std::cerr << "{graph.name} tiling failed in lambda" << std::endl;
            return -1;
        }}
    }}

    at::Tensor workspace_tensor;
    void *workspace = nullptr;
    if (workspace_size > 0) {{
        workspace_tensor = AllocateWorkspaceTensor(workspace_size, current_stream);
        workspace = const_cast<void *>(workspace_tensor.storage().data());
        if (workspace == nullptr) {{
            std::cerr << "{graph.name} allocate workspace failed" << std::endl;
            return -1;
        }}
    }}
    DLOG() << "Launch args for {graph.name}: block_dim=" << block_dim << " stream=" << current_stream
            << " workspace_size=" << workspace_size << " workspace=" << workspace << std::endl;

    {buffer_assign_in_taskqueue1}

    auto launch_call = [=]() -> int {{
        int64_t inner_result = launch_fn({', '.join(["block_dim", "current_stream"] + launch_args + ["const_cast<AutofuseTilingData*>(&tiling_data)"])});
        if (inner_result != 0) {{
            std::cerr << "{graph.name} launch failed" << std::endl;
            return -1;
        }}
        return 0;
    }};

    at_npu::native::OpCommand::RunOpApiV2("{graph.name}", launch_call);
    return 0;
}}
    ''')

    # taskqueue=2:静态tiling在queue外执行，动态tiling+workspace+kernel在queue内执行
    else:
        wrapper.splice(f'''
extern "C" int wrapper({signature}) {{
    if (tiling_fn == nullptr || launch_fn == nullptr) {{
        if (tiling_fn == nullptr) std::cerr << "{graph.name} tiling func null" << std::endl;
        if (launch_fn == nullptr) std::cerr << "{graph.name} launch func null" << std::endl;
        return -1;
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        std::cerr << "{graph.name} GetStream failed in wrapper thread" << std::endl;
        return -1;
    }}

    auto launch_call = [=]() -> int {{
        uint32_t workspace_size = default_workspace_size;
        uint32_t block_dim = default_block_dim;
        {tiling_dtype} tiling_data = default_tiling_data;
        if (!static_tiling) {{
            int64_t tiling_result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"])});
            if (tiling_result != 0) {{
                std::cerr << "{graph.name} tiling failed in lambda" << std::endl;
                return -1;
            }}
        }}

        at::Tensor workspace_tensor;
        void *workspace = nullptr;
        if (workspace_size > 0) {{
            workspace_tensor = AllocateWorkspaceTensor(workspace_size, current_stream);
            workspace = const_cast<void *>(workspace_tensor.storage().data());
            if (workspace == nullptr) {{
                std::cerr << "{graph.name} allocate workspace failed" << std::endl;
                return -1;
            }}
        }}
        DLOG() << "Launch args for {graph.name}: block_dim=" << block_dim << " stream=" << current_stream
               << " workspace_size=" << workspace_size << " workspace=" << workspace << std::endl;

        {buffer_assign_in_taskqueue2}

        int64_t inner_result = launch_fn({', '.join(["block_dim", "current_stream"] + launch_args + ["&tiling_data"])});
        if (inner_result != 0) {{
            std::cerr << "{graph.name} launch failed" << std::endl;
            return -1;
        }}
        return 0;
    }};

    at_npu::native::OpCommand::RunOpApiV2("{graph.name}", launch_call);
    return 0;
}}
    ''')

    return wrapper.getvalue()
