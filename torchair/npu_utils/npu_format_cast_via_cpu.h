#ifndef TORCHAIR_TORCHAIR_NPU_UTILS_NPU_FORMAT_CAST_VIA_CPU_H_
#define TORCHAIR_TORCHAIR_NPU_UTILS_NPU_FORMAT_CAST_VIA_CPU_H_


#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>

namespace tng {
namespace npu_utils {
at::Tensor npu_format_cast_via_cpu(
    const at::Tensor& self,
    int64_t acl_format,
    c10::optional<int64_t> customize_dtype,
    c10::optional<int64_t> input_dtype
);
}
}
#endif  // TORCHAIR_TORCHAIR_NPU_UTILS_NPU_FORMAT_CAST_VIA_CPU_H_
