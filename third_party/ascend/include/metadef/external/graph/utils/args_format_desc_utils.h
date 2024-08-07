/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* ===================================================================================================================*/

#ifndef METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
#define METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H

#include <string>
#include <vector>

#include "graph/ge_error_codes.h"
#include "graph/op_desc.h"
#include "register/hidden_input_func_registry.h"

namespace ge {
enum class AddrType {
  INPUT = 0,
  OUTPUT,
  WORKSPACE,
  TILING,
  INPUT_DESC,
  OUTPUT_DESC,
  FFTS_ADDR,
  OVERFLOW_ADDR,
  TILING_FFTS,
  HIDDEN_INPUT,
  TILING_CONTEXT,
  OP_TYPE,
  PLACEHOLDER
};

enum class TilingContextSubType { TILING_CONTEXT = -1, TILING_DATA, TILING_KEY, BLOCK_DIM };

struct ArgDesc {
  AddrType addr_type;
  int32_t ir_idx;
  bool folded;
  uint8_t reserved[4];
};

class ArgsFormatDescUtils {
 public:
  // i* -> ir_idx = -1, folded=false
  // 对于输入输出，idx表示ir定义的idx，-1表示所有输入、所有输出，此时非动态输入、输出默认展开,动态输出要i1*这样才表示展开
  // 对于workspace -1表示个数未知，folded暂时无意义
  // 对ffts尾块非尾块地址，idx=0表示非尾块，idx=1表示尾块
  // 对于其他类型, idx和fold 暂时没有意义
  static void Append(std::vector<ArgDesc> &arg_descs, AddrType type, int32_t ir_idx = -1, bool folded = false);

  static void AppendHiddenInput(std::vector<ArgDesc> &arg_descs, HiddenInputType hidden_type);

  static void AppendTilingContext(std::vector<ArgDesc> &arg_descs,
                                  TilingContextSubType sub_type = TilingContextSubType::TILING_CONTEXT);

  static std::string ToString(const std::vector<ArgDesc> &arg_descs);

  // 字符串用i*这样的通配符时，返回的argDesc不会按照实际个数展开
  static graphStatus Parse(const std::string &str, std::vector<ArgDesc> &arg_descs);

  static std::string Serialize(const std::vector<ArgDesc> &arg_descs);
};
}

#endif  // METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
