// 相关头文件放最前，便于检查其自包含性
#include "npu_format_cast_via_cpu.h"

#include <cstdio>
#include <string>

#include "acl/acl_rt.h"
#include "torch/csrc/Exceptions.h"
#include "torch_npu/inc/core/NPUBridge.h"
#include "torch_npu/inc/core/NPUFormat.h"
#include "torch_npu/inc/core/NPUStream.h"

namespace tng {
namespace npu_utils {

struct TngRuntimeError : public torch::PyTorchError {
    using PyTorchError::PyTorchError;
    PyObject *python_type() override {
        return PyExc_RuntimeError;
    }
};

template <typename T>
constexpr T ceil_div(T a, T b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
std::vector<T> trans_nd_to_nz(const std::vector<T> &weight_array, uint64_t k, uint64_t n, uint64_t c0_size) {
    uint64_t block_size = 16;
    uint64_t k1 = ceil_div(k, block_size);
    uint64_t n1 = ceil_div(n, c0_size);
    uint64_t batch = weight_array.size() / (k * n);
    uint64_t weight_nz_size = batch * n1 * k1 * block_size * c0_size;
    std::vector<T> weight_nz_array(weight_nz_size, T{});

    // (k, n) -> (n1, k1, k0, n0)
    // (k, n) -> (ceil_div(n, c0), ceil_div(k, 16), 16, c0)
    for (size_t out_idx = 0; out_idx < weight_array.size(); ++out_idx) {
        size_t idx_b = out_idx / (k * n);
        size_t idx = out_idx % (k * n);
        size_t idx_k = idx / n;
        size_t idx_n = idx % n;
        size_t idx_k0 = idx_k % block_size;
        size_t idx_k1 = idx_k / block_size;
        size_t idx_n0 = idx_n % c0_size;
        size_t idx_n1 = idx_n / c0_size;
        size_t new_idx = idx_b * n1 * k1 * block_size * c0_size + idx_n1 * k1 * block_size * c0_size +
                         idx_k1 * block_size * c0_size + idx_k0 * c0_size + idx_n0;
        weight_nz_array[new_idx] = weight_array[out_idx];
    }
    return weight_nz_array;
}

int32_t GetStreamTimeout() {
    char *env_val = std::getenv("ACL_STREAM_TIMEOUT");
    int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
    return static_cast<int32_t>(envFlag);
}

#define TNG_RAISE_ASSERT(expr, msg, ...)                                             \
    do {                                                                             \
        const auto &status = (expr);                                                 \
        if (!status) {                                                               \
            char _tng_assert_buf[512];                                               \
            std::snprintf(_tng_assert_buf, sizeof(_tng_assert_buf), (msg), ##__VA_ARGS__); \
            throw TngRuntimeError(_tng_assert_buf);                                  \
        }                                                                            \
    } while (false)

/*
* @brief 在CPU上将输入张量转换为指定格式，目前支持 int8/uint8 的输入，输出为 int8/uint8 c0=32 FRACTAL_NZ 格式
* @param self 输入张量
* @param acl_format 输出格式，目前支持 29，即 int8/uint8 c0=32 FRACTAL_NZ 格式
* @param customize_dtype 自定义数据类型，目前不支持
* @param input_dtype 输入数据类型，目前不支持
* @return 输出张量
*/
at::Tensor npu_format_cast_via_cpu(
    const at::Tensor &self,
    int64_t acl_format,
    c10::optional<int64_t> customize_dtype,
    c10::optional<int64_t> input_dtype) {
    // 当前支持的输入组合：
    // | self.dtype()   | acl_format | customize_dtype | input_dtype    | output_dtype                |
    // |----------------|------------|-----------------|----------------|-----------------------------|
    // | int8/uint8     | 29         | 任意            | 任意           | int8/uint8 c0=32 FRACTAL_NZ  |
    uint64_t c0_size = 32;
    aclFormat output_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    TNG_RAISE_ASSERT(acl_format == 29, "acl_format must be 29, but is %ld", acl_format);
    at::Tensor self_cpu = self.to("cpu");
    self_cpu = self_cpu.to(at::kByte);
    auto nd_dim_num = self.dim();
    auto nd_first_dim = self.size(nd_dim_num - 2);
    auto nd_last_dim = self.size(nd_dim_num - 1);
    std::vector<uint8_t> nd_data(self_cpu.data_ptr<uint8_t>(), self_cpu.data_ptr<uint8_t>() + self_cpu.numel());

    // 仅支持 int8/uint8；customize_dtype 和 input_dtype 不受限制（任意值均可）。
    bool is_supported_dtype = (self.dtype() == at::kByte || self.dtype() == at::kChar);
    std::string dtype_str(self.dtype().name());
    TNG_RAISE_ASSERT(
        is_supported_dtype,
        "unsupported dtype %s, and input_dtype %ld",
        dtype_str.c_str(),
        input_dtype.has_value() ? static_cast<int64_t>(input_dtype.value()) : static_cast<int64_t>(-1));

    std::vector<uint8_t> nz_data = trans_nd_to_nz(nd_data, nd_first_dim, nd_last_dim, c0_size);

    c10::TensorOptions options = self.options();
    at::Tensor result = at_npu::native::empty_with_format(self.sizes(), options, output_format);

    int64_t nbytes = static_cast<int64_t>(nz_data.size()) * sizeof(uint8_t);
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    int32_t timeout = GetStreamTimeout();

    TNG_RAISE_ASSERT(
        aclrtSynchronizeStreamWithTimeout(stream, timeout) == ACL_SUCCESS,
        "aclrtSynchronizeStreamWithTimeout failed");
    TNG_RAISE_ASSERT(
        aclrtMemcpy(const_cast<void *>(result.storage().unsafeGetStorageImpl()->data()), nbytes,
                    nz_data.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        "aclrtMemcpy failed");
    result.set_(result.storage(), 0, self.sizes(), self.strides());
    TNG_RAISE_ASSERT(result.defined(), "result.set_ failed");

    return result;
}
}  // namespace npu_utils
}  // namespace tng
