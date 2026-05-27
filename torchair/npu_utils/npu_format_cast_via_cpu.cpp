// 相关头文件放最前，便于检查其自包含性
#include "npu_format_cast_via_cpu.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "securec.h"

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

namespace {
constexpr uint64_t kBlockSize = 16;
constexpr uint64_t kInt8NzC0Size = 32;
constexpr int64_t kSupportedAclFormat = 29;
constexpr size_t kAssertMsgBufSize = 512;
}  // namespace

#define TNG_RAISE_ASSERT(expr, msg, ...)                                             \
    do {                                                                             \
        const auto &status = (expr);                                                 \
        if (!status) {                                                               \
            char _tng_assert_buf[kAssertMsgBufSize];                                 \
            int _tng_assert_ret = snprintf_s(                                        \
                _tng_assert_buf, sizeof(_tng_assert_buf), sizeof(_tng_assert_buf) - 1, (msg), ##__VA_ARGS__); \
            if (_tng_assert_ret < 0) {                                              \
                throw TngRuntimeError("format assert message failed");              \
            }                                                                        \
            throw TngRuntimeError(_tng_assert_buf);                                  \
        }                                                                            \
    } while (false)

template <typename T>
T ceil_div(T a, T b) {
    TNG_RAISE_ASSERT(
        b != 0,
        "invalid divisor for ceil_div, dividend=%llu, divisor=%llu",
        static_cast<unsigned long long>(a),
        static_cast<unsigned long long>(b));
    return (a + b - 1) / b;
}

template <typename T>
std::vector<T> trans_nd_to_nz(const std::vector<T> &weight_array, uint64_t k, uint64_t n, uint64_t c0_size) {
    TNG_RAISE_ASSERT(
        k != 0 && n != 0 && c0_size != 0,
        "invalid shape for trans_nd_to_nz, k=%llu, n=%llu, c0_size=%llu",
        static_cast<unsigned long long>(k),
        static_cast<unsigned long long>(n),
        static_cast<unsigned long long>(c0_size));

    uint64_t k1 = ceil_div(k, kBlockSize);
    uint64_t n1 = ceil_div(n, c0_size);
    uint64_t batch = weight_array.size() / (k * n);
    uint64_t weight_nz_size = batch * n1 * k1 * kBlockSize * c0_size;
    std::vector<T> weight_nz_array(weight_nz_size, T{});

    // (k, n) -> (n1, k1, k0, n0)
    // (k, n) -> (ceil_div(n, c0), ceil_div(k, 16), 16, c0)
    for (size_t out_idx = 0; out_idx < weight_array.size(); ++out_idx) {
        size_t idx_b = out_idx / (k * n);
        size_t idx = out_idx % (k * n);
        size_t idx_k = idx / n;
        size_t idx_n = idx % n;
        size_t idx_k0 = idx_k % kBlockSize;
        size_t idx_k1 = idx_k / kBlockSize;
        size_t idx_n0 = idx_n % c0_size;
        size_t idx_n1 = idx_n / c0_size;
        size_t new_idx = idx_b * n1 * k1 * kBlockSize * c0_size + idx_n1 * k1 * kBlockSize * c0_size +
                         idx_k1 * kBlockSize * c0_size + idx_k0 * c0_size + idx_n0;
        weight_nz_array[new_idx] = weight_array[out_idx];
    }
    return weight_nz_array;
}

int32_t GetStreamTimeout() {
    char *env_val = std::getenv("ACL_STREAM_TIMEOUT");
    if (env_val == nullptr) {
        return -1;
    }

    errno = 0;
    char *end = nullptr;
    long timeout = std::strtol(env_val, &end, 10);
    if (errno == ERANGE || end == env_val || *end != '\0') {
        throw TngRuntimeError("ACL_STREAM_TIMEOUT must be a valid int32 value");
    }
    if (timeout < std::numeric_limits<int32_t>::min() || timeout > std::numeric_limits<int32_t>::max()) {
        throw TngRuntimeError("ACL_STREAM_TIMEOUT out of int32 range");
    }
    return static_cast<int32_t>(timeout);
}

/*
* @brief 在CPU上将输入张量转换为指定格式，目前支持 int8 的输入，输出为 int8 c0=32 FRACTAL_NZ 格式
* @param self 输入张量
* @param acl_format 输出格式，目前支持 29，即 int8 c0=32 FRACTAL_NZ 格式
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
    // | int8           | 29         | 任意            | 任意           | int8 c0=32 FRACTAL_NZ        |
    const uint64_t c0_size = kInt8NzC0Size;
    const aclFormat output_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    TNG_RAISE_ASSERT(
        acl_format == kSupportedAclFormat,
        "acl_format must be %lld, but is %lld",
        static_cast<long long>(kSupportedAclFormat),
        static_cast<long long>(acl_format));
    TNG_RAISE_ASSERT(self.defined(), "input tensor must be defined");
    auto nd_dim_num = self.dim();
    TNG_RAISE_ASSERT(
        nd_dim_num >= 2,
        "input tensor dim must be at least 2, but is %lld",
        static_cast<long long>(nd_dim_num));
    auto nd_first_dim = self.size(nd_dim_num - 2);
    auto nd_last_dim = self.size(nd_dim_num - 1);
    TNG_RAISE_ASSERT(
        nd_first_dim > 0 && nd_last_dim > 0,
        "input tensor last two dims must be greater than 0, but are %lld and %lld",
        static_cast<long long>(nd_first_dim),
        static_cast<long long>(nd_last_dim));

    // 仅支持 int8；customize_dtype 和 input_dtype 不受限制（任意值均可）。
    bool is_supported_dtype = (self.dtype() == at::kChar);
    std::string dtype_str(self.dtype().name());
    TNG_RAISE_ASSERT(
        is_supported_dtype,
        "unsupported dtype %s, and input_dtype %lld",
        dtype_str.c_str(),
        input_dtype.has_value() ? static_cast<long long>(input_dtype.value()) : static_cast<long long>(-1));

    at::Tensor self_cpu = self.to("cpu").to(at::kByte);
    std::vector<uint8_t> nd_data(self_cpu.data_ptr<uint8_t>(), self_cpu.data_ptr<uint8_t>() + self_cpu.numel());

    std::vector<uint8_t> nz_data = trans_nd_to_nz(nd_data, nd_first_dim, nd_last_dim, c0_size);

    c10::TensorOptions options = self.options();
    at::Tensor result = at_npu::native::empty_with_format(self.sizes(), options, output_format);
    auto result_storage = result.storage();

    TNG_RAISE_ASSERT(
        nz_data.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "NZ size exceeds int64 max");
    const int64_t nbytes = static_cast<int64_t>(nz_data.size()) * static_cast<int64_t>(sizeof(uint8_t));
    const uint64_t allocated_size = static_cast<uint64_t>(result_storage.nbytes());
    TNG_RAISE_ASSERT(
        static_cast<uint64_t>(nbytes) <= allocated_size,
        "NZ size %lld exceeds allocated size %llu",
        static_cast<long long>(nbytes),
        static_cast<unsigned long long>(allocated_size));
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    int32_t timeout = GetStreamTimeout();

    TNG_RAISE_ASSERT(
        aclrtSynchronizeStreamWithTimeout(stream, timeout) == ACL_SUCCESS,
        "aclrtSynchronizeStreamWithTimeout failed");
    auto *storage_impl = result_storage.unsafeGetStorageImpl();
    TNG_RAISE_ASSERT(storage_impl != nullptr, "storage_impl is null");
    const void *result_data_ptr = storage_impl->data();
    TNG_RAISE_ASSERT(result_data_ptr != nullptr, "result data_ptr is null");
    TNG_RAISE_ASSERT(
        aclrtMemcpy(const_cast<void *>(result_data_ptr), nbytes,
                    nz_data.data(), nbytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        "aclrtMemcpy failed");
    result.set_(result_storage, 0, self.sizes(), self.strides());
    TNG_RAISE_ASSERT(result.defined(), "result.set_ failed");

    return result;
}
}  // namespace npu_utils
}  // namespace tng
