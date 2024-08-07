/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <map>
#include <tuple>
#include <vector>
#include <mutex>
#include "nlohmann/json.hpp"
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"

#ifndef INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_
#define INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_

namespace nnopbase {
// 二进制内容，用于内存中的二进制.o文件描述
struct Binary {
  const uint8_t *content; // 二进制内容的起始指针
  uint32_t len; // 二进制的长度
};

class OpBinaryResourceManager {
public:
  static OpBinaryResourceManager &GetInstance()
  {
    static OpBinaryResourceManager manager;
    return manager;
  }

  ~OpBinaryResourceManager() = default;

  void AddOpFuncHandle(const ge::AscendString &opType, const std::vector<void *> &opResourceHandle);
  ge::graphStatus AddBinary(const ge::AscendString &opType,
                            const std::vector<std::tuple<const uint8_t*, const uint8_t*>> &opBinary);
  ge::graphStatus AddRuntimeKB(const ge::AscendString &opType,
                               const std::vector<std::tuple<const uint8_t*, const uint8_t*>> &opRuntimeKb);

// 获取资源
public:
  // 获取所有算子的描述信息
  const std::map<const ge::AscendString, nlohmann::json> &GetAllOpBinaryDesc() const;

  // 获取某个算子的描述信息
  ge::graphStatus GetOpBinaryDesc(const ge::AscendString &opType, nlohmann::json &binDesc) const;

  // 根据json文件路径(算子json中存在)查找.json/.o的信息
  ge::graphStatus GetOpBinaryDescByPath(const ge::AscendString &jsonFilePath,
                                        std::tuple<nlohmann::json, Binary> &binInfo) const;

  // 根据simplifiedKey(算子json中存在)查找.json/.o的信息
  ge::graphStatus GetOpBinaryDescByKey(const ge::AscendString &simplifiedKey,
                                       std::tuple<nlohmann::json, Binary> &binInfo) const;

  // 二进制知识库
  ge::graphStatus GetOpRuntimeKB(const ge::AscendString &opType, std::vector<ge::AscendString> &kbList) const;

private:
  OpBinaryResourceManager() = default; // 单例，禁止外部创建对象
  OpBinaryResourceManager &operator=(const OpBinaryResourceManager &) = delete; // 禁止拷贝
  OpBinaryResourceManager &operator=(OpBinaryResourceManager &&) = delete;
  OpBinaryResourceManager(const OpBinaryResourceManager &) = delete;
  OpBinaryResourceManager(OpBinaryResourceManager &&) = delete;

  mutable std::recursive_mutex mutex_;

  // 二进制描述信息 opType -> xxx.json
  std::map<const ge::AscendString, nlohmann::json> opBinaryDesc_;

  // 二进制jsonPath simplifiedKey -> jsonPath, 可能存在多个simplifiedKey对应同一个jsonPath
  std::map<const ge::AscendString, ge::AscendString> keyToPath_;

  // 二进制信息 jsonPath -> xxx.json, xxx.o
  std::map<const ge::AscendString, std::tuple<nlohmann::json, Binary>> pathToBinary_;

  // 二进制知识库 opType -> xxx1.json, xxx2.json
  std::map<const ge::AscendString, std::vector<ge::AscendString>> runtimeKb_;

  // infershape/op tiling/runtime kb parser等全局变量指针，注册类资源，仅需持有
  std::map<const ge::AscendString, std::vector<void *>> resourceHandle_;
};
} // nnopbase

#endif  // INC_EXTERNAL_REGISTER_OP_BINARY_RESOURCE_MANAGER_H_
