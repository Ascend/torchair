/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_TILING_INFO_H_
#define INC_EXTERNAL_REGISTER_OP_TILING_INFO_H_

#include <sstream>
#include <map>
#include <memory>
#include "external/graph/ge_error_codes.h"
#include "external/graph/ascend_string.h"
#include "external/graph/tensor.h"

namespace optiling {
using ByteBuffer = std::stringstream;

enum class TensorArgType {
  TA_NONE,
  TA_SINGLE,
  TA_LIST,
};

class TeOpVarAttrArgsImpl;
class TeOpVarAttrArgs {
  friend class VarAttrHelper;

public:
  TeOpVarAttrArgs() = default;
  ~TeOpVarAttrArgs() = default;
  const uint8_t *GetData(const std::string &name, const std::string &dtype, size_t &size) const;

private:
  std::shared_ptr<TeOpVarAttrArgsImpl> impl_;
};

struct TeOpTensor {
  std::vector<int64_t> shape;
  std::vector<int64_t> ori_shape;
  std::string format;
  std::string ori_format;
  std::string dtype;
  std::string name;
  std::map<std::string, std::string> attrs;
};

struct TeOpTensorArg {
  TensorArgType arg_type;
  std::vector<TeOpTensor> tensor;
};

struct OpRunInfo {
  uint32_t block_dim;
  std::vector<int64_t> workspaces;
  ByteBuffer tiling_data;
  bool clear_atomic;
  uint64_t tiling_key;
  int32_t tiling_cond;
};

using TeOpAttrArgs = std::vector<std::string>;
using TeConstTensorData = std::tuple<const uint8_t *, size_t, ge::Tensor>;

struct TeOpParas {
  std::vector<TeOpTensorArg> inputs;
  std::vector<TeOpTensorArg> outputs;
  std::map<std::string, TeConstTensorData> const_inputs;
  TeOpAttrArgs attrs;
  std::string op_type;
  TeOpVarAttrArgs var_attrs;
};

struct OpCompileInfo {
  std::string str;
  std::string key;
};

namespace utils {
class OpRunInfoImpl;
class OpRunInfo {
public:
  OpRunInfo();
  ~OpRunInfo() = default;

  OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key);
  // Copy
  OpRunInfo(const OpRunInfo &runinfo);
  // Move
  OpRunInfo(OpRunInfo &&runinfo);
  // Copy
  OpRunInfo &operator=(const OpRunInfo &runinfo);
  // Move
  OpRunInfo &operator=(OpRunInfo &&runinfo);

  void SetBlockDim(const uint32_t &block_dim);
  uint32_t GetBlockDim() const;
  void SetScheduleMode(const uint32_t schedule_mode);
  uint32_t GetScheduleMode() const;
  void AddWorkspace(const int64_t &workspace);
  size_t GetWorkspaceNum() const;
  ge::graphStatus GetWorkspace(const size_t &idx, int64_t &workspace) const;
  void GetAllWorkspaces(std::vector<int64_t> &workspaces) const;
  const std::vector<int64_t> &GetAllWorkspaces() const;
  void SetWorkspaces(const std::vector<int64_t> &workspaces);

  template<class T>
  void AddTilingData(const T &value) {
    AddTilingData(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  template <typename T>
  void operator << (const T &value) {
    AddTilingData(reinterpret_cast<const char *>(&value), sizeof(T));
  }
  void AddTilingData(const char *value, const size_t size);
  void* GetAddrBase(uint64_t& max_size) const;
  void SetAddrBaseOffset(const uint64_t size);
  ByteBuffer &GetAllTilingData();
  const ByteBuffer &GetAllTilingData() const;
  void InternelSetTiling(const ByteBuffer &value);
  void SetClearAtomic(const bool clear_atomic);
  bool GetClearAtomic() const;

  void SetTilingKey(const uint64_t &new_tiling_key);
  uint64_t GetTilingKey() const;
  uint64_t GetTilingDataSize() const;
  void ResetWorkspace();
  void ResetAddrBase(void *const addr_base, const uint64_t max_size);
  void AlignOffsetWith64();
  bool SetMemCheckBaseOffset(const uint64_t &offset);
  void SetTilingCond(const int32_t tiling_cond);
  int32_t GetTilingCond() const;
private:
  std::shared_ptr<OpRunInfoImpl> impl_;
};

class OpCompileInfoImpl;
class OpCompileInfo {
public:
  OpCompileInfo();
  ~OpCompileInfo() = default;
  OpCompileInfo(const ge::AscendString &key, const ge::AscendString &value);
  OpCompileInfo(const std::string &key, const std::string &value);
  // Copy
  OpCompileInfo(const OpCompileInfo &compileinfo);
  // Move
  OpCompileInfo(OpCompileInfo &&compileinfo);
  // Copy
  OpCompileInfo &operator=(const OpCompileInfo &compileinfo);
  // Move
  OpCompileInfo &operator=(OpCompileInfo &&compileinfo);

  void SetKey(const ge::AscendString &key);
  const ge::AscendString &GetKey() const;

  void SetValue(const ge::AscendString &value);
  const ge::AscendString &GetValue() const;

private:
  std::shared_ptr<OpCompileInfoImpl> impl_;
};
}
}  // namespace optiling
#endif  // INC_REGISTER_OP_TILING_REGISTRY_H_
