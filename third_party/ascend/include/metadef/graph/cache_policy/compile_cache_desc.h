/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef GRAPH_CACHE_POLICY_COMPILE_CACHE_DESC_H
#define GRAPH_CACHE_POLICY_COMPILE_CACHE_DESC_H

#include <string>
#include <vector>
#include "cache_desc.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"
#include "graph/types.h"
#include "graph/def_types.h"
#include "graph/hash_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/debug/log.h"

namespace ge {
class CompileCacheDesc;
using CompileCacheDescPtr = std::shared_ptr<CompileCacheDesc>;
class BinaryHolder {
 public:
  BinaryHolder() = default;

  ~BinaryHolder() = default;

  BinaryHolder(const BinaryHolder &other);
  BinaryHolder(BinaryHolder &&other);
  BinaryHolder &operator=(const BinaryHolder &other);
  BinaryHolder &operator=(BinaryHolder &&other);

  BinaryHolder(const uint8_t *const data, const size_t data_len);

  static std::unique_ptr<BinaryHolder> createFrom(std::unique_ptr<uint8_t[]> &&ptr, size_t length);

  const uint8_t *GetDataPtr() const noexcept;

  const size_t &GetDataLen() const noexcept;

  bool operator!=(const BinaryHolder &second) const;

 private:
  std::unique_ptr<uint8_t[]> holder_ = nullptr;
  size_t data_len_ = 0UL;
};

class TensorInfoArgs {
 public:
  TensorInfoArgs(const Format format, const Format origin_format, const DataType data_type)
    : format_(format),
      origin_format_(origin_format),
      data_type_(data_type) {}

  ~TensorInfoArgs() = default;

  bool IsUnknownShape() const;
  bool IsShapeInRange(const TensorInfoArgs &other) const;
  bool IsTensorInfoMatch(const TensorInfoArgs &other) const;
  Format GetFormat() const;
  Format GetOriginFormat() const;
  DataType GetDataType() const;
  void SetShape(const std::vector<int64_t> &shape);
  void SetShape(const SmallVector<int64_t, kDefaultDimsNum> &shape);
  void SetOriginShape(const std::vector<int64_t> &origin_shape);
  void SetOriginShape(const SmallVector<int64_t, kDefaultDimsNum> &origin_shape);
  void SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges);
  bool operator!=(const TensorInfoArgs &second) const;

 private:
  Format format_;
  Format origin_format_;
  DataType data_type_;
  SmallVector<int64_t, kDefaultMaxInputNum> shape_;
  SmallVector<int64_t, kDefaultMaxInputNum> origin_shape_;
  SmallVector<std::pair<int64_t, int64_t>, kDefaultMaxInputNum> shape_range_;
};

class CompileCacheDesc : public CacheDesc {
  friend class CacheHasher;
 public:
  CompileCacheDesc() = default;
  ~CompileCacheDesc() = default;
  bool IsEqual(const CacheDescPtr &other) const override;
  bool IsMatch(const CacheDescPtr &other) const override;
  CacheHashKey GetCacheDescHash() const override;
  void SetOpType(const std::string &op_type);
  void AddBinary(const BinaryHolder &holder);
  void AddBinary(BinaryHolder &&holder);
  void AddTensorInfo(const TensorInfoArgs &tensor_info);
  void SetScopeId(const std::initializer_list<uint64_t> scope_id);
  size_t GetTensorInfoSize();
  TensorInfoArgs *MutableTensorInfo(size_t index);

 private:
  bool CheckWithoutTensorInfo(const CompileCacheDesc *first, const CompileCacheDesc *second) const;
  std::string op_type_; // op type
  SmallVector<uint64_t, kDefaultMaxInputNum> scope_id_; // graph_id and session_id
  SmallVector<TensorInfoArgs, kDefaultMaxInputNum> tensor_info_args_vec_; // input tensordescs
  SmallVector<BinaryHolder, kDefaultMaxInputNum> other_desc_; // attrs float float size
};
}  // namespace ge

namespace std {
template<>
struct hash<ge::BinaryHolder> {
  size_t operator()(const ge::BinaryHolder &value) const {
    GE_CHECK_NOTNULL(value.GetDataPtr());
    size_t seed = ge::HashUtils::MultiHash();
    const uint64_t u8_data = ge::PtrToValue(ge::PtrToPtr<const uint8_t, const void>(value.GetDataPtr()));
    for (size_t idx = 0UL; idx < value.GetDataLen(); idx++) {
      seed = ge::HashUtils::HashCombine(seed, *(ge::PtrToPtr<void, uint8_t>(ge::ValueToPtr(u8_data + idx))));
    }
    return seed;
  }
};
}  // namespace std
#endif
