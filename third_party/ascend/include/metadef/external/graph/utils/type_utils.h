/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_GRAPH_UTILS_TYPE_UTILS_H_
#define INC_GRAPH_UTILS_TYPE_UTILS_H_

#include <map>
#include <set>
#include <unordered_set>
#include <string>
#include "graph/types.h"

namespace ge {
class TypeUtils {
 public:
  static std::string DataTypeToSerialString(const DataType data_type);
  static DataType SerialStringToDataType(const std::string &str);
  static std::string FormatToSerialString(const Format format);
  static Format SerialStringToFormat(const std::string &str);
  static Format DataFormatToFormat(const std::string &str);
  static bool GetDataTypeLength(const ge::DataType data_type, uint32_t &length);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TYPE_UTILS_H_
