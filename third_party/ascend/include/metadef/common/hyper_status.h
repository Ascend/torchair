/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
#define AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
#include <memory>
#include <cstdarg>
#include "graph/types.h"
namespace gert {
ge::char_t *CreateMessage(const ge::char_t *format, va_list arg);
class HyperStatus {
 public:
  bool IsSuccess() const {
    return status_ == nullptr;
  }
  const ge::char_t *GetErrorMessage() const noexcept {
    return status_;
  }
  ~HyperStatus() {
    delete[] status_;
  }

  HyperStatus() {}
  HyperStatus(const HyperStatus &other);
  HyperStatus(HyperStatus &&other) noexcept;
  HyperStatus &operator=(const HyperStatus &other);
  HyperStatus &operator=(HyperStatus &&other) noexcept;

  static HyperStatus Success();
  static HyperStatus ErrorStatus(const ge::char_t *message, ...);
  static HyperStatus ErrorStatus(std::unique_ptr<ge::char_t[]> message);

 private:
  ge::char_t *status_ = nullptr;
};
}  // namespace gert

namespace ge {
using HyperStatus = gert::HyperStatus;
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_HYPER_STATUS_H_
