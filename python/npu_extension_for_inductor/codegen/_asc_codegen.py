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
    graph_py_code.splice(f"from autofuse.pyautofuse import ascir")
    graph_py_code.splice(f'from autofuse.pyautofuse import Autofuser, AutofuserOptions')
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
    tiling_signature.append(f"uint32_t *workspace_size")
    tiling_signature.append(f"uint32_t *block_dim")
    tiling_signature.append(f"void *resource_limit")
    pgo_tiling_signature = ["char* file"] + tiling_signature

    workspaces = [TensorArg("workspace")]
    launch_args = [v.name for v in itertools.chain(inputs, outputs, workspaces)]
    launch_signature = ["uint32_t block_dim", "void *stream"]
    launch_signature.extend([v.signature for v in itertools.chain(inputs, outputs, workspaces)])
    launch_signature.append(f"{tiling_dtype} *tiling_data")

    pgo_signature = [v.signature for v in itertools.chain(inputs, outputs)]
    pgo_signature.extend(["void *stream"])

    wrapper.splice(f'''
typedef int64_t (*TilingFuncType)({', '.join(tiling_signature)});
typedef int64_t (*LaunchFuncType)({', '.join(launch_signature)});
typedef int64_t (*PGOTilingFuncType)({', '.join(pgo_tiling_signature)});
static TilingFuncType tiling_fn = reinterpret_cast<TilingFuncType>(GetFunc("AutofuseTiling"));
static LaunchFuncType launch_fn = reinterpret_cast<LaunchFuncType>(GetFunc("AutofuseLaunch"));
static PGOTilingFuncType pgo_tiling_fn = reinterpret_cast<PGOTilingFuncType>(GetFunc("PgoAutofuseTiling"));
const static bool pgo_enable = std::getenv("EXPERIMENTAL_AUTOFUSE_PGO") != nullptr;

extern "C" int wrapper({signature}) {{
    {tiling_dtype} tiling_data;
    uint32_t workspace_size = 0;
    uint32_t block_dim = 0;
    int64_t result = 0;
    if (tiling_fn == nullptr || launch_fn == nullptr || ((pgo_tiling_fn == nullptr) && pgo_enable)) {{
        if (tiling_fn == nullptr) std::cerr << "{graph.name} kernel tiling func not found" << std::endl;
        if (launch_fn == nullptr) std::cerr << "{graph.name} kernel launch func not found" << std::endl;
        if (pgo_tiling_fn == nullptr) std::cerr << "{graph.name} kernel pgo tiling func not found" << std::endl;
        return -1;
    }}
    if (pgo_enable) {{
        result = pgo_tiling_fn((char *)config_file, {', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"])});
    }} else {{
        result = tiling_fn({', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"])});
    }}
    if (result != 0) {{
        std::cerr << "{graph.name} kernel tiling failed" << std::endl;
        return -1;
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        std::cerr << "{graph.name} kernel get stream failed" << std::endl;
        return -1;
    }}

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

extern "C" int wrapper_only_launch({', '.join(pgo_signature)}, uint32_t workspace_size, {tiling_dtype} *tiling_data) {{
    uint32_t block_dim = tiling_data->block_dim;
    if (launch_fn == nullptr) {{
        std::cerr << "{graph.name} kernel launch func not found" << std::endl;
        return -1;
    }}

    if (tiling_data == nullptr) {{
        std::cerr << "{graph.name} tiling data is null" << std::endl;
    }}

    void *current_stream = GetStream(stream);
    if (current_stream == nullptr) {{
        std::cerr << "{graph.name} kernel get stream failed" << std::endl;
        return -1;
    }}

    void *workspace = nullptr;
    if (workspace_size > 0) {{
        workspace = MallocWorkspace(workspace_size, current_stream);
        if (workspace == nullptr) {{
            std::cerr << "{graph.name} kernel malloc workspace failed" << std::endl;
            return -1;
        }}
    }}

    int64_t result = launch_fn({', '.join(["block_dim", "current_stream"] + launch_args + ["tiling_data"])});
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


def codegen_pgo_def(graph: FusedASCGraph):
    pgo = IndentedBuffer()
    inputs = [TensorArg(v) for v in graph.inputs]
    outputs = [TensorArg(v) for v in graph.outputs]
    symbols = [SymArg(str(v)) for v in graph.size_vars]
    stream = StreamArg("stream")
    tiling_dtype = f"AutofuseTilingData"

    all_args = [TensorArg(v) for v in graph.args] + symbols + [stream]
    signature = ', '.join([v.signature for v in all_args])

    buffer_assign_pgo = ''
    for in_name, out_name in zip(graph.inputs + graph.outputs, graph.inputs_outer + graph.outputs_outer):
        buffer_assign_pgo += f'\n    auto *{in_name} = {out_name};'

    tiling_args = [v.name for v in symbols]
    tiling_signature = [v.signature for v in symbols]
    tiling_signature.append(f"{tiling_dtype} *tiling_data")
    tiling_signature.append(f"uint32_t *workspace_size")
    tiling_signature.append(f"uint32_t *block_dim")
    tiling_signature.append(f"void *resource_limit")

    pgo_launch_args = [v.name for v in itertools.chain(inputs, outputs)]
    pgo_signature = [v.signature for v in itertools.chain(inputs, outputs)]
    pgo_signature.extend(["void *stream"])
    pgo_tiling_signature = (["char* search_file", "char* config_file"] + tiling_signature
        + [v.signature for v in itertools.chain(inputs, outputs)]
        + ["void *stream", "void *prof_callback", "void *prof_batch_callback"])


    pgo.splice(f'''
typedef int64_t (*PgoTilingSearchType)({', '.join(pgo_tiling_signature)});
typedef int64_t (*WrapperOnlyLaunchFuncType)({', '.join(pgo_signature)}, uint32_t workspace_size, {tiling_dtype} *tiling_data);
static PgoTilingSearchType pgo_search_fn = reinterpret_cast<PgoTilingSearchType>(GetFunc("PgoTilingSearch"));
static WrapperOnlyLaunchFuncType wrapper_fn = reinterpret_cast<WrapperOnlyLaunchFuncType>(GetWrapperFunc("wrapper_only_launch"));

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)         \\\\
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static size_t group_size = 1000;
static std::map<uint64_t, msptiActivity*> g_profiling_map;
static uint64_t loop = 10;
static int max_flush_times = 5;
static double best_perf = DBL_MAX;

static const char* GetActivityKindString(msptiActivityKind kind) {{
    static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {{
        {{MSPTI_ACTIVITY_KIND_INVALID, "INVALID"}},
        {{MSPTI_ACTIVITY_KIND_MARKER, "MARKER"}},
        {{MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"}},
        {{MSPTI_ACTIVITY_KIND_API, "API"}},
        {{MSPTI_ACTIVITY_KIND_HCCL, "HCCL"}},
        {{MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"}},
        {{MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"}},
        {{MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"}},
        {{MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}}
    }};
    auto it = STRING_MAP.find(kind);
    return it != STRING_MAP.end() ? it->second : "<unknown>";
}}

static const char* GetResultCodeString(msptiResult result) {{
    static const std::unordered_map<msptiResult, const char*> STRING_MAP = {{
        {{MSPTI_SUCCESS, "SUCCESS"}},
        {{MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"}},
        {{MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"}},
        {{MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"}},
        {{MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"}},
        {{MSPTI_ERROR_INNER, "ERROR_INNER"}}
    }};

    auto it = STRING_MAP.find(result);
    return it != STRING_MAP.end() ? it->second : "<unknown>";
}}

void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {{
    DLOG() << "[mspti] UserBufferRequest..." << std::endl;
    uint8_t *pBuffer = reinterpret_cast<uint8_t *>(malloc(16 * 1024 * 1024 + ALIGN_SIZE));
    *buffer = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
    *size = 16 * 1024 * 1024;
    *maxNumRecords = 0;
}}

void UserBufferComplete(uint8_t *buffer, size_t size, size_t validSize) {{
    DLOG() << "[mspti] UserBufferComplete, address: " << reinterpret_cast<uintptr_t>(buffer) << ", size: " << size 
    << ", valid size: " << validSize << std::endl;
    if (validSize > 0) {{
        msptiActivity *pRecord = NULL;
        msptiResult status = MSPTI_SUCCESS;
        do {{
            status = msptiActivityGetNextRecord(buffer, validSize, &pRecord);
            if (status == MSPTI_SUCCESS) {{
                if (pRecord->kind == MSPTI_ACTIVITY_KIND_KERNEL) {{
                    msptiActivityKernel* kernelRecord = (msptiActivityKernel*)pRecord;
                    msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
                    memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
                    memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
                    g_profiling_map[kernelRecord->start] = pRecordCopy;

                }} else {{
                    DLOG() << "[mspti] [" << GetActivityKindString(pRecord->kind) << "] ignored" << std::endl;
                }}
            }} else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {{
                break;
            }} else {{
                DLOG() << "[mspti] Consume data fail error is" << GetResultCodeString(status) << std::endl;
                break;
            }}
        }} while (1);
    }}
    free(buffer);
}}

void SetUpMspti(msptiSubscriberHandle* subscriber) {{
    DLOG() << "[mspti] setup mspti" << std::endl;
    msptiSubscribe(subscriber, nullptr, nullptr);
    msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
    msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
}}

void TearDownMspti(msptiSubscriberHandle* subscriber) {{
    DLOG() << "[mspti] tear down mspti" << std::endl;
    msptiUnsubscribe(*subscriber);
    msptiActivityFlushAll(1);
}}

int ProfilingBatchProcess({', '.join(pgo_signature)}, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, std::vector<AutofuseTilingDataPerf>::iterator end) {{
    uint64_t batch_size = end - begin;
    g_profiling_map.clear();
    msptiSubscriberHandle subscriber;
    SetUpMspti(&subscriber);
    
    static int64_t count = 0;
    count++;
    
    int64_t result = 0;
    for (auto it = begin; it != end; ++it) {{
        it->best_perf = DBL_MAX;
        {tiling_dtype} tilingData = it->tiling_data;
        for (uint64_t i = 0; i < loop; ++i) {{
            result = wrapper_fn({', '.join(pgo_launch_args + ["stream", "workspace_size", "&tilingData"])});
            if (result != 0) {{
                std::cerr << "[PGO] {graph.name} ProfilingBatchProcess launch failed loop:" << i << std::endl;
                TearDownMspti(&subscriber);
                return -1;
            }}
        }}
    }}
    
    result = aclrtSynchronizeStream(stream);
    if (result != 0) {{
        std::cerr << "[PGO] {graph.name} ProfilingBatchProcess sync stream failed" << std::endl;
        TearDownMspti(&subscriber);
        return -1;
    }}
    TearDownMspti(&subscriber);
    
    int flush_count = 0;
    while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {{
        flush_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
        msptiActivityFlushAll(1);
    }}
    
    if (g_profiling_map.size() != batch_size * loop) {{
        std::cerr << "[PGO] {graph.name} map size " << g_profiling_map.size() << " not equals to loop * batch_size " << batch_size * loop << std::endl;
        return -1;
    }}
    
    auto it = g_profiling_map.begin();
    for (uint64_t i = 0; i < batch_size; ++i) {{
        int total_duration = 0;
        for (uint64_t j = 0; j < loop; ++j) {{
            msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);
            total_duration += kernel->end - kernel->start;
            std::advance(it, 1);
        }}
        double average_duration = static_cast<double>(total_duration) / loop;
        (begin + i)->best_perf = average_duration;
        if (best_perf > average_duration) {{
            best_perf = average_duration;
        }}
        DLOG() << "average_duration:" << average_duration << " best_perf:" << best_perf << " count:" << count << " batch_size:" << batch_size << " flush_count:" << flush_count << std::endl;
    }}
    return 0;
}}

extern "C" long int PGOGetProfilingBatch({', '.join(pgo_signature)}, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {{
    int case_num = profiles->size();
    DLOG() << "{graph.name} PGOGetProfilingBatch case_num:" << case_num << std::endl;
    int64_t result = 0;
    auto it = profiles->begin();
    while (it != profiles->end()) {{
        auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;
        size_t start_index = std::distance(profiles->begin(), it);
        for (int i = 0; i < 3; i++) {{
            result = ProfilingBatchProcess({', '.join(pgo_launch_args + ["stream", "workspace_size"])}, it, end_it);
            if (result != 0) {{
                std::cerr << "[PGO] {graph.name} ProfilingBatchProcess failed at start_index:" << start_index << " retry time:" << i << std::endl;
            }} else {{
                break;
            }}
        }}
        it = end_it;
    }}
    return 0;
}}


extern "C" long int PGOGetProfiling({', '.join(pgo_signature)}, uint32_t workspace_size, {tiling_dtype}* tilingData, double* outCostTime) {{
    g_profiling_map.clear();
    msptiSubscriberHandle subscriber;
    SetUpMspti(&subscriber);

    int64_t result = -1;
    *outCostTime = DBL_MAX;
    static int64_t count = 0;
    count++;

    for (uint64_t j = 0; j < loop; ++j) {{
        result = wrapper_fn({', '.join(pgo_launch_args + ["stream", "workspace_size", "tilingData"])});
        if (result != 0) {{
            std::cerr << "[PGO] {graph.name} launch failed loop:" << j << std::endl;
            TearDownMspti(&subscriber);
            return -1;
        }}
    }}

    result = aclrtSynchronizeStream(stream);
    if (result != 0) {{
        std::cerr << "[PGO] {graph.name} sync stream failed" << std::endl;
        TearDownMspti(&subscriber);
        return -1;
    }}
    TearDownMspti(&subscriber);

    int flush_count = 0;
    while (g_profiling_map.size() < loop && flush_count < max_flush_times) {{
      flush_count++;
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
      msptiActivityFlushAll(1);
    }}

    if (g_profiling_map.size() != loop) {{
        std::cerr << "[PGO] {graph.name} map size " << g_profiling_map.size() << " not equals to loop " << loop << std::endl;
        return -1;
    }}

    int total_duration = 0;
    for (const auto& pair : g_profiling_map) {{
        msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);
        total_duration += kernel->end - kernel->start;
    }}
    double average_duration = static_cast<double>(total_duration) / loop;
    *outCostTime = average_duration;

    if (best_perf > *outCostTime) {{
        best_perf = *outCostTime;
    }}
    DLOG() << "average_duration:" << *outCostTime << " best_perf:" << best_perf << " count:" << count << " flush_count:" << flush_count << std::endl;
    return 0;
}}

extern "C" int pgo({signature}) {{
    {tiling_dtype} tiling_data = {{0}};
    uint32_t workspace_size = 0;
    uint32_t block_dim = 0;
    if (pgo_search_fn == nullptr || wrapper_fn == nullptr) {{
        if (pgo_search_fn == nullptr) std::cerr << "[PGO] {graph.name} pgo search func not found" << std::endl;
        if (wrapper_fn == nullptr) std::cerr << "[PGO] {graph.name} wrapper func not found" << std::endl;
        return -1;
    }}
    
    stream = GetStream(stream);
    if (stream == nullptr) {{
        std::cerr << "{graph.name} kernel get stream failed" << std::endl;
        return -1;
    }}

    {buffer_assign_pgo}

    int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, {', '.join(tiling_args + ["&tiling_data", "&workspace_size", "&block_dim", "nullptr"] + pgo_launch_args + ["stream", "reinterpret_cast<void*>(PGOGetProfiling)", "reinterpret_cast<void*>(PGOGetProfilingBatch)"])});
    if (result != 0) {{
        std::cerr << "[PGO] {graph.name} mspti profiling failed" << std::endl;
        return -1;
    }}

    return 0;
}}

    ''')

    return pgo.getvalue()