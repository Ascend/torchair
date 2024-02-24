/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef INC_GRAPH_OP_SO_BIN_H
#define INC_GRAPH_OP_SO_BIN_H

#include <string>
#include <utility>
#include "graph/types.h"
#include "graph/def_types.h"

namespace ge {
typedef struct {
  std::string cpu_info;
  std::string os_info;
  std::string opp_version;
  std::string compiler_version;
} SoInOmInfo;

class OpSoBin {
public:
  OpSoBin(const std::string &so_name, const std::string &vendor_name,
      std::unique_ptr<char_t[]> data, uint32_t data_len)
      : so_name_(so_name), vendor_name_(vendor_name), data_(std::move(data)),
        data_size_(data_len) {}

  ~OpSoBin() = default;

  const std::string &GetSoName() const { return so_name_; }
  const std::string &GetVendorName() const { return vendor_name_; }
  const uint8_t *GetBinData() const { return ge::PtrToPtr<void, const uint8_t>(data_.get()); }
  size_t GetBinDataSize() const { return data_size_; }
  OpSoBin(const OpSoBin &) = delete;
  const OpSoBin &operator=(const OpSoBin &) = delete;

private:
  std::string so_name_;
  std::string vendor_name_;
  std::unique_ptr<char_t[]> data_;
  uint32_t data_size_;
};

using OpSoBinPtr = std::shared_ptr<ge::OpSoBin>;
}  // namespace ge

#endif  // INC_GRAPH_OP_SO_BIN_H
