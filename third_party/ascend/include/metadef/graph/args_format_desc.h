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

#ifndef METADEF_CXX_ARGS_FORMAT_H
#define METADEF_CXX_ARGS_FORMAT_H

#include <string>
#include <vector>

#include "common/ge_common/debug/ge_log.h"
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
  PLACEHOLDER
};

struct ArgDesc {
  AddrType addr_type;
  int32_t ir_idx;
  bool folded;
  uint8_t reserved[4];
};

class ArgsFormatDesc {
 public:
  // i* -> ir_idx = -1, folded=false
  // 对于输入输出，idx表示ir定义的idx，-1表示所有输入、所有输出，此时非动态输入、输出默认展开,动态输出要i1*这样才表示展开
  // 对于workspace -1表示个数未知，folded暂时无意义
  // 对ffts尾块非尾块地址，idx=0表示非尾块，idx=1表示尾块
  // 对于其他类型, idx和fold 暂时没有意义
  void Append(AddrType type, int32_t ir_idx = -1, bool folded = false);

  void AppendHiddenInput(HiddenInputType hidden_type);

  std::string ToString() const;

  graphStatus GetArgsSize(const OpDescPtr &op_desc, size_t &args_size) const;

  // 为了方便使用，字符串用i*这样的通配符时，返回的argDesc会按照实际个数展开
  static graphStatus Parse(const OpDescPtr &op_desc, const std::string &str, std::vector<ArgDesc> &arg_descs);

  static std::string Serialize(const std::vector<ArgDesc> &arg_descs);

 private:
  std::vector<ArgDesc> arg_descs_;
};
}  // namespace ge

#endif  // METADEF_CXX_ARGS_FORMAT_H
