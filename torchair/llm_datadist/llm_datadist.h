#ifndef TORCH_AIR_TORCH_AIR_CORE_LLM_DATADIST_H_
#define TORCH_AIR_TORCH_AIR_CORE_LLM_DATADIST_H_

#include <utility>
#include <vector>
#include "torch/torch.h"

namespace llm_datadist {
// llm datadist
enum class TorchDataType : int32_t {
  kBool,
  kInt8,
  kUint8,
  kInt16,
  kInt32,
  kInt64,
  kFloat16,
  kBfloat16,
  kFloat32,
  kFloat64,
  kComplex32,
  kComplex64,
  kComplex128,
};

std::pair<uint32_t, std::vector<at::Tensor>> AsTorchTensor(const std::vector<int64_t> &dims, const int32_t ge_data_type,
                                                           const std::vector<uintptr_t> &addresses);
}  // namespace llm_datadist

#endif  // TORCH_AIR_TORCH_AIR_CORE_LLM_DATADIST_H_