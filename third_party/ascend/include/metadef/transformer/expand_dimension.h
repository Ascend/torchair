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

#ifndef COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
#define COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_

#include <string>
#include <vector>
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "exe_graph/runtime/shape.h"

namespace transformer {
/* Pad dimension according to reshape type */
bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, ge::GeShape &shape);

bool ExpandRangeDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                          const uint32_t &tensor_index, const std::string &reshape_type,
                          std::vector<std::pair<int64_t, int64_t>> &ranges);

class ExpandDimension {
public:
  ExpandDimension();
  ~ExpandDimension();

  static int64_t GenerateReshapeType(const ge::Format &origin_format, const ge::Format &format,
                                     const size_t &origin_dim_size, const std::string &reshape_type);
  static bool GenerateReshapeType(const ge::Format &origin_format, const ge::Format &format,
                                  const size_t &origin_dim_size, const std::string &reshape_type,
                                  int64_t &reshape_type_mask);
  static bool GenerateReshapeTypeByMask(const ge::Format &origin_format, const size_t &origin_dim_size,
                                        const int64_t &reshape_type_mask, std::string &reshape_type,
                                        std::string &failed_reason);
  static void ExpandDims(const int64_t &reshape_type, ge::GeShape &shape);
  static void ExpandDims(const int64_t &reshape_type, const ge::GeShape &origin_shape, ge::GeShape &shape);
  static void ExpandDims(const int64_t &reshape_type, gert::Shape &shape);
  static void ExpandDims(const int64_t &reshape_type, const gert::Shape &origin_shape, gert::Shape &shape);
  static bool GetDefaultReshapeType(const ge::Format &origin_format, const size_t &origin_dim_size,
                                    std::string &reshape_type);
  static int32_t GetAxisIndexByName(char ch, const ge::Format &format);
  static int64_t GetReshapeAxicValue(const int64_t &reshape_type_mask,
                                     const ge::GeShape &shape, int32_t axis_index);
  static int64_t GetReshapeAxicValueByName(const int64_t &reshape_type_mask, char ch,
                                           const ge::GeShape &shape, const ge::Format &format);
private:
  static bool GetFormatFullSize(const ge::Format &format, size_t &full_size);
  static bool IsNeedExpand(const ge::Format &origin_format, const ge::Format &format,
                           const size_t &origin_dim_size, const size_t &full_size, const std::string &reshape_type);
  static bool IsReshapeTypeValid(const ge::Format &origin_format, const size_t &origin_dim_size,
                                 const std::string &reshape_type);
};
} // namespace transformer
#endif //COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
