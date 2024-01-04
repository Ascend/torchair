import ctypes
import os
import json
import re
import itertools
import tempfile
from _ctypes import Structure, byref
from ctypes import cdll, c_size_t, c_int64, c_void_p
import subprocess

import torch
from torch._inductor.codegen.common import IndentedBuffer
from npu_extension_for_inductor.common.op_code import OpCode, OpProto
from npu_extension_for_inductor.common.utils import TypeUtils


def _camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def debug(func):
    def wrapper(*args, **kwargs):
        buf = IndentedBuffer()
        buf.writeline(f'#ifdef INDUCTOR_DEBUG')
        ret = func(*args, **kwargs)
        buf.splice(ret)
        buf.writeline(f'#endif')
        return buf

    return wrapper


class AclnnArg:
    def __init__(self, name):
        self.name = name
        self.trace = None
        self.inner = name

    def __call__(self, *args):
        if not self.trace:
            self.trace = f"{self.name}->"
        self.trace = f"{self.trace}({', '.join(args)})"
        return self

    def __getattr__(self, item):
        if not self.trace:
            self.trace = f"{self.name}->{item}"
        else:
            self.trace = f"{self.trace}.{item}"
        return self

    def __str__(self):
        v = self.trace
        self.trace = None
        return v

    def pre(self, code: IndentedBuffer):
        pass

    def post(self, code: IndentedBuffer):
        pass


class TensorArg(AclnnArg):
    def __init__(self, name):
        super().__init__(name)
        self.inner = f"{name}Tensor"
        self.signature = f"Tensor *{name}"

    def pre(self, code: IndentedBuffer):
        code.writeline(f"auto {self.name}Addr = {self.data_ptr};")
        code.splice(f"""
        #ifdef CPU_ONLY
        auto {self.name}Size = GetTensorSize({self.name});
        ASSERT_ACL_SUCCESS(aclrtMalloc(&{self.name}Addr, {self.name}Size, ACL_MEM_MALLOC_NORMAL_ONLY));
        ASSERT_ACL_SUCCESS(aclrtMemcpy({self.name}Addr, {self.name}Size, {self.data_ptr}, {self.name}Size, ACL_MEMCPY_HOST_TO_DEVICE));
        #endif
        """)
        code.writeline(
            f"auto {self.inner} = aclCreateTensor({self.shape.dims}, {self.shape.dim_num}, aclDataType({self.dtype}), nullptr, 0, ACL_FORMAT_ND, {self.shape.dims}, {self.shape.dim_num}, {self.name}Addr);")
        code.splice(self.debug())

    def post(self, code: IndentedBuffer):
        code.splice(f"""
        #ifdef CPU_ONLY
        ASSERT_ACL_SUCCESS(aclrtMemcpy({self.data_ptr}, {self.name}Size, {self.name}Addr, {self.name}Size, ACL_MEMCPY_DEVICE_TO_HOST));
        ASSERT_ACL_SUCCESS(aclrtFree({self.name}Addr));
        #endif
        """)

    @debug
    def debug(self):
        buf = IndentedBuffer()
        buf.splice(f'std::cerr << "Input:{self.name} = " << DebugString({self.name}) << std::endl;')
        return buf


class StreamArg(AclnnArg):
    def __init__(self, name):
        super().__init__(name)
        self.signature = f"aclrtStream {name}"

    def pre(self, code: IndentedBuffer):
        code.splice(self.debug())

    @debug
    def debug(self):
        buf = IndentedBuffer()
        buf.writeline(f'std::cerr << "Input:{self.name} = Stream(addr=" << {self.inner} << ")" << std::endl;')
        return buf


class SymValsArg(AclnnArg):
    def __init__(self, name):
        super().__init__(name)
        self.inner = f"{name}Tensor"
        self.signature = f"SymVals *{name}"

    def pre(self, code: IndentedBuffer):
        code.writeline(
            f"auto {self.inner} = aclCreateTensor({self.vals}, {self.num}, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, {self.vals}, {self.num}, nullptr);")
        code.splice(self.debug())

    @debug
    def debug(self):
        buf = IndentedBuffer()
        buf.writeline(f'std::cerr << "Input:{self.name} = SymVals(" << DebugString({self.name}) << ")" << std::endl;')
        return buf


class CSymVals(Structure):
    _fields_ = [("num", c_size_t), ("vals", c_void_p)]

    C_DEFINE = """
struct SymVals {
    size_t num = 0;
    int64_t *vals = nullptr;
};"""


class CTensor(Structure):
    class Shape(Structure):
        MAX_DIM_NUM = 24
        _fields_ = [("dim_num", c_size_t), ("dims", c_int64 * MAX_DIM_NUM)]

    _fields_ = [("data_ptr", c_size_t), ("shape", Shape), ("dtype", c_int64)]

    C_DEFINE = """
static constexpr size_t MAX_DIM_NUM = 24;
struct Tensor {
    void *data_ptr = nullptr;
    struct Shape {
        size_t dim_num;
        int64_t dims[MAX_DIM_NUM];
    } shape;
    int32_t dtype = ACL_DT_UNDEFINED;
};"""


class AclnnKernelBin:
    def __init__(self, proto: OpProto):
        self.name = proto.name
        self.lib_file = f'{_camel_to_snake(self.name)}.so'

        self.header_file = None
        self.inputs = [TensorArg(v.name) for v in proto.inputs if v.name != "size_vars"]
        self.outputs = [TensorArg(v.name) for v in proto.outputs]
        self.sym_vals = SymValsArg("sym_vals")
        self.stream = StreamArg("stream")
        self.all_args = [self.sym_vals] + self.inputs + self.outputs + [self.stream]

        code = IndentedBuffer()
        signature = ', '.join([v.signature for v in self.all_args])
        signature = f'extern "C" aclError wrapper({signature})'
        code.writeline(signature)
        code.writeline('{')
        with code.indent():
            for arg in self.all_args:
                arg.pre(code)

            workspace_args = ', '.join([v.inner for v in itertools.chain([self.sym_vals], self.inputs, self.outputs)])
            code.splice(f"""
                uint64_t workspace_size = 0U;
                void *workspace = nullptr;
                aclOpExecutor *handle = nullptr;
                ASSERT_ACL_SUCCESS(aclnn{self.name}GetWorkspaceSize({workspace_args}, &workspace_size, &handle));
                ASSERT_ACL_SUCCESS(aclnn{self.name}(workspace, workspace_size, handle, {self.stream.inner}));
                #ifdef CPU_ONLY
                ASSERT_ACL_SUCCESS(aclrtSynchronizeStream({self.stream.inner}));
                #endif
            """)

            for arg in self.all_args:
                arg.post(code)

            code.writeline("return ACL_SUCCESS;")

        code.writeline('}')
        self.kernel_code = code

    def compile(self, *, cwd, compile_flags):
        wrapper = IndentedBuffer()
        wrapper.writeline(f'#include "acl/acl.h"')
        wrapper.writeline(f'#include "aclnn_{_camel_to_snake(self.name)}.h"')
        wrapper.writeline(f'#include <iostream>')
        wrapper.writeline(f'#include <sstream>')
        wrapper.writeline(f'#include <vector>')
        wrapper.splice(CTensor.C_DEFINE)
        wrapper.splice(CSymVals.C_DEFINE)

        wrapper.splice("""
        #ifdef INDUCTOR_DEBUG
            #define LOG_ACL_SUCCESS(expr) std::cerr << "ACL_SUCCESS:" << (expr) << std::endl;
            std::string DebugString(Tensor *t) {
                std::stringstream ss;
                ss << "Tensor(dtype=" << t->dtype << ", shape=";
                if (t->shape.dim_num == 0U) {
                    ss << "[]";
                } else {
                    ss << "[" << t->shape.dims[0];
                    for (size_t i = 1; i < t->shape.dim_num; i++) {
                        ss << ", " << t->shape.dims[i];
                    }
                    ss << "]";
                }
                ss << ", addr=" << t->data_ptr << ")" << std::endl;
                return ss.str();
            }
            std::string DebugString(SymVals *sym_vals) {
                std::stringstream ss;
                if (sym_vals->num > 0U) {
                    int64_t *vals = static_cast<int64_t *>(sym_vals->vals);
                    ss << "s0=" << vals[0];
                    for (size_t i = 1; i < sym_vals->num; i++) {
                        ss << ", s" << i << "=" << vals[i];
                    }
                }
                return ss.str();
            }
        #else
            #define LOG_ACL_SUCCESS(expr)
        #endif

        #define ASSERT_ACL_SUCCESS(expr) \\
            do { \\
                auto ret = (expr); \\
                if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) { \\
                    auto err = aclGetRecentErrMsg(); \\
                    std::cerr << "ACL_ERROR:" << ret << ":" << (err == nullptr ? "unknown" : err) << std::endl; \\
                    return ret; \\
                } \\
                else { \\
                    LOG_ACL_SUCCESS(#expr); \\
                } \\
            } while (0)
        """)

        # TODO: remove this once inherit into torch_npu
        wrapper.splice("""
        #ifdef CPU_ONLY
        static auto _ = [](){
            ASSERT_ACL_SUCCESS(aclInit(nullptr));
            ASSERT_ACL_SUCCESS(aclrtSetDevice(0));
            return ACL_SUCCESS;
        }();

        size_t GetTensorSize(Tensor *t) {
            const static std::vector<size_t> kTypeSize = [](){
                std::vector<size_t> type_size;
                type_size.resize(ACL_COMPLEX32 + 1, 0);
                type_size[ACL_FLOAT16] = 2;
                type_size[ACL_FLOAT] = 4;
                type_size[ACL_DOUBLE] = 8;
                type_size[ACL_INT8] = 1;
                type_size[ACL_UINT8] = 1;
                type_size[ACL_INT32] = 4;
                type_size[ACL_INT64] = 8;
                type_size[ACL_BOOL] = 1;
                type_size[ACL_BF16] = 2;
                return type_size;
            }();
            if (t->dtype < 0 || t->dtype >= kTypeSize.size()) {
                return 0U;
            }
            size_t size = kTypeSize[t->dtype];
            for (size_t i = 0; i < t->shape.dim_num; i++) {
                size *= t->shape.dims[i];
            }
            const static size_t kAlignSize = 32U;
            return (size + kAlignSize - 1) / kAlignSize * kAlignSize;
        }
        #endif
        """)

        wrapper.splice(self.kernel_code)

        os.makedirs(cwd, exist_ok=True)
        kernel_file = os.path.join(cwd, "kernel.cpp")
        with open(kernel_file, 'w') as f:
            f.write(wrapper.getvalue())

        args = ["g++", "-shared", "-fPIC", "-DINDUCTOR_DEBUG", "-DCPU_ONLY", "-o", self.lib_file,
                kernel_file] + compile_flags
        print(' '.join(args), flush=True)
        subprocess.run(args, cwd=cwd, check=True)

        return os.path.join(cwd, self.lib_file)


class AclnnPrj:
    def __init__(self, root: str, src: OpCode):
        self.root = os.path.join(root, "aclnn")
        self.kernel_path = os.path.join(root, "kernel")
        self.host = os.path.join(self.root, "op_host")
        self.device = os.path.join(self.root, "op_kernel")
        self.src = src
        self.proto = self.src.proto
        self.vendor = f"{self.proto.name}"
        self.opp_path = os.getenv("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp")
        self.ascend_path = os.path.dirname(self.opp_path)

    def create(self, *, core_type):
        os.makedirs(self.root, exist_ok=True)
        op_name = self.proto.name
        op_name_snake = _camel_to_snake(op_name)
        config_path = os.path.join(self.root, f"{op_name}.json")
        with open(config_path, 'w') as f:
            f.write("[")
            json.dump(self.proto.json, f, indent=4)
            f.write("]")
        args = ["msopgen", "gen", "-i", config_path, "-lan", "cpp", "-c", core_type, "-out", self.root]
        print(' '.join(args), flush=True)
        subprocess.run(args, check=True)

        assert os.path.exists(self.host)
        assert os.path.exists(self.device)

        cmake_presets = os.path.join(self.root, "CMakePresets.json")
        assert os.path.exists(cmake_presets)
        with open(cmake_presets, 'r') as f:
            cmake_presets_json = json.load(f)
        cmake_presets_json["configurePresets"][0]["cacheVariables"]["ASCEND_CANN_PACKAGE_PATH"][
            "value"] = self.ascend_path
        cmake_presets_json["configurePresets"][0]["cacheVariables"]["vendor_name"]["value"] = self.vendor
        with open(cmake_presets, 'w') as f:
            json.dump(cmake_presets_json, f)

        tiling_file = os.path.join(self.host, f"{op_name_snake}_tiling.h")
        op_host_file = os.path.join(self.host, f"{op_name_snake}.cpp")
        op_kernel_file = os.path.join(self.device, f"{op_name_snake}.cpp")

        for f, content in [(tiling_file, self.src.tiling), (op_host_file, self.src.host),
                           (op_kernel_file, self.src.device)]:
            with open(f, 'w') as f:
                f.write(content)

    def load_kernel(self, *, env: dict = None, cwd=None):
        env = env or {}
        args = [f"{k}={v}" for k, v in env.items()] + ['./build.sh']
        subprocess.run(args, cwd=self.root, check=True)

        import glob
        pkgs = glob.glob(os.path.join(self.root, "build_out", "*.run"))
        assert len(pkgs) == 1, str(pkgs)
        subprocess.run([f"{pkgs[0]}"], cwd=os.path.join(self.root, "build_out"), check=True)

        compile_flags = [f"-I{self.ascend_path}/include"]
        compile_flags.append(f"-I{self.opp_path}/vendors/{self.vendor}/op_api/include")
        compile_flags.append(f"-L{self.ascend_path}/lib64")
        compile_flags.append(f"-Wl,-rpath,{self.opp_path}/vendors/{self.vendor}/op_api/lib")
        compile_flags.append(f"-L{self.opp_path}/vendors/{self.vendor}/op_api/lib")
        compile_flags.append(f"-lcust_opapi")
        compile_flags.append(f"-lascendcl")

        kernel = AclnnKernelBin(self.proto)
        kernel_bin = kernel.compile(cwd=self.kernel_path, compile_flags=compile_flags)
        return AclnnKernel(kernel_bin)


class AclnnKernel:
    def __init__(self, kernel_bin: str):
        lib = cdll.LoadLibrary(kernel_bin)
        self.kernel = getattr(lib, f"wrapper")

    def __call__(self, *tensors: CTensor, sym_vals: CSymVals):
        args = [byref(sym_vals)]
        args += [byref(tensor) for tensor in tensors]
        self.kernel(*args, None)  # No stream specified from python


class NpuInductorKernel:
    def __init__(self, aclnn_kernel: AclnnKernel):
        self.kernel = aclnn_kernel

    def __call__(self, *args: torch.Tensor, sym_vals):
        aclnn_args = []
        for arg in args:
            assert arg.is_contiguous()
            aclnn_args.append(CTensor(data_ptr=arg.data_ptr(), shape=CTensor.Shape(dim_num=arg.dim(), dims=(arg.shape)),
                                      dtype=TypeUtils.torch_to_acl(arg.dtype)))
        vals = ctypes.cast((c_int64 * len(sym_vals))(*sym_vals), c_void_p)
        self.kernel(*aclnn_args, sym_vals=CSymVals(num=c_size_t(len(sym_vals)), vals=vals))


_compile_cache = dict()


def _compile(src: OpCode):
    prj = AclnnPrj(tempfile.mkdtemp(), src)
    core_type = os.getenv("NPU_CORE_TYPE", "ai_core-ascend910B1")
    prj.create(core_type=core_type)
    aclnn_kernel = prj.load_kernel(env={})
    kernel = NpuInductorKernel(aclnn_kernel)
    _compile_cache[src.proto.name] = kernel
    return kernel


def compile(src: OpCode):
    kernel = _compile_cache.get(src.proto.name) or _compile(src)
    return kernel