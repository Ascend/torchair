/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef METADEF_INC_COMMON_SCREEN_PRINTER_H_
#define METADEF_INC_COMMON_SCREEN_PRINTER_H_

#include <mutex>
#include <string>
#include <unordered_set>
#include <unistd.h>
#include "external/graph/types.h"

namespace ge {
class ScreenPrinter {
 public:
  static ScreenPrinter &GetInstance();
  void Log(const char *fmt, ...);
  void Init(const std::string &print_mode);

 private:
  ScreenPrinter() = default;
  ~ScreenPrinter() = default;

  ScreenPrinter(const ScreenPrinter &) = delete;
  ScreenPrinter(const ScreenPrinter &&) = delete;
  ScreenPrinter &operator=(const ScreenPrinter &)& = delete;
  ScreenPrinter &operator=(const ScreenPrinter &&)& = delete;

  enum class PrintMode : uint32_t {
    ENABLE = 0U,
    DISABLE = 1U
  };
  PrintMode print_mode_ = PrintMode::ENABLE;
  std::mutex mutex_;
};

#define SCREEN_LOG(fmt, ...)                                       \
  do {                                                             \
    ScreenPrinter::GetInstance().Log(fmt, ##__VA_ARGS__);          \
  } while (false)
}  // namespace ge
#endif  // METADEF_INC_COMMON_SCREEN_PRINTER_H_
