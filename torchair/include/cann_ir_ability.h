#ifndef TORCH_AIR_TORCH_AIR_CORE_CANN_IR_ABILITY_H_
#define TORCH_AIR_TORCH_AIR_CORE_CANN_IR_ABILITY_H_

#include <utility>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "torch/torch.h"

namespace cann_ir_ability {
std::string CheckCannCompat(const std::string &optype, const std::vector<std::string> &optional_inputs,
                            const std::vector<std::string> &optional_attrs) noexcept;
}  // namespace cann_ir_ability

#endif  // TORCH_AIR_TORCH_AIR_CORE_CANN_IR_ABILITY_H_
