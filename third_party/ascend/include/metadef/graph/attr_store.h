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

#ifndef EXECUTE_GRAPH_ATTR_STORE_H
#define EXECUTE_GRAPH_ATTR_STORE_H
#include <string>
#include <unordered_map>
#include <map>
#include <set>

#include "any_value.h"

namespace ge {
using AttrId = uint64_t;
using AttrSubId = uint32_t;
enum class AttrType : uint32_t {
  kAttrPredefinedInIr = 0U,  // IR预定义的属性
  kAttrGeneral = 1U,         // 通用属性
  kAttrTypeEnd = 2U
};
constexpr inline uint32_t GetAttrType(const AttrId id) {
  return static_cast<uint32_t>(id >> 32U);
}
constexpr inline uint32_t GetSubAttrId(const AttrId id) {
  return static_cast<uint32_t>(id & 0xffffffffU);
}
constexpr inline AttrId GetAttrId(const uint32_t type, const uint32_t sub_id) {
  return (static_cast<uint64_t>(type) << 32U) | static_cast<uint64_t>(sub_id);
}

class AttrStore {
 public:
  static AttrStore Create(const size_t pre_defined_attr_count);

  template<typename T>
  bool Set(const AttrId attr_id, T &&value) const;
  template<typename T>
  bool Set(const AttrId attr_id, const T &value) const;
  template<typename T>
  bool SetByName(const std::string &name, T &&value);
  template<typename T>
  bool SetByName(const std::string &name, const T &value);

  template<typename T>
  const T *Get(const AttrId attr_id) const;
  template<typename T>
  T *MutableGet(const AttrId attr_id);
  template<typename T>
  const T *GetByName(const std::string &name) const;
  template<typename T>
  T *MutableGetByName(const std::string &name);

  AttrId GetIdByName(const std::string &name) const noexcept;
  void SetNameAndId(std::string name, const AttrId id);

  bool Exists(const AttrId attr_id) const noexcept;
  bool Exists(const std::string &name) const noexcept;

  bool Delete(const std::string &name);
  void Clear();

  void Swap(AttrStore &other);
  bool SetAnyValueByName(const std::string &name, const AnyValue &value);

  // unordered版本更好，为了兼容老版本接口，仍然用set和map，不论用哪种数据结构，这都是非常低效的接口
  std::set<std::string> GetAllAttrNames() const;
  std::map<std::string, AnyValue> GetAllAttrs() const;

  AnyValue *MutableAnyValue(const std::string &name) const noexcept;
  AnyValue *GetOrCreateAnyValue(const std::string &name);
  const AnyValue *GetAnyValue(const std::string &name) const noexcept;

 private:
  AnyValue *MutableAnyValue(const AttrId attr_id) const noexcept;
  AnyValue *GetOrCreateAnyValue(const AttrId attr_id) const;
  const AnyValue *GetAnyValue(const AttrId attr_id) const noexcept;

  class PreDefinedAttrStore {
  public:
    bool Exists(const AttrSubId index) const noexcept;
    bool Delete(const AttrSubId index);
    void Clear();
    void Swap(PreDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(const AttrSubId index) const;
    AnyValue *MutableAnyValue(const AttrSubId index) const noexcept;
    const AnyValue *GetAnyValue(const AttrSubId index) const noexcept;

    void Resize(const size_t s);

   private:
    std::vector<AnyValue> attrs_;
  };

  class CustomDefinedAttrStore {
   public:
    bool Exists(const std::string &name) const noexcept;
    bool Delete(const std::string &name);
    void Clear();
    void Swap(CustomDefinedAttrStore &other);

    AnyValue *GetOrCreateAnyValue(const std::string &name);
    AnyValue *MutableAnyValue(const std::string &name) const noexcept;
    const AnyValue *GetAnyValue(const std::string &name) const noexcept;

    void GetAllNames(std::set<std::string> &names) const;
    void GetAllAttrs(std::map<std::string, AnyValue> &names_to_attr) const;

   private:
    std::unordered_map<std::string, AnyValue> attrs_;
  };

  std::unordered_map<std::string, AttrId> names_to_id_;
  // 更好的办法是定义一个虚基类、派生出两个子类，然后保存两个子类的指针：`std::array<std::unique_ptr<SubAttrStore>, kAttrTypeEnd>`
  // 然后根据不同的SubAttr类型，调用对应子类的函数。但是这么做会导致创建AttrStore时，总会带有两次子类实例堆申请的开销，
  // 为了减少堆内存申请，直接将子类平铺在成员变量上。
  PreDefinedAttrStore pre_defined_attrs_;
  CustomDefinedAttrStore general_attrs_;
};

template<typename T>
bool AttrStore::Set(const AttrId attr_id, const T &value) const {
  auto *const v = GetOrCreateAnyValue(attr_id);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(value);
  return true;
}
template<typename T>
bool AttrStore::Set(const AttrId attr_id, T &&value) const {
  auto *const v = GetOrCreateAnyValue(attr_id);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(std::forward<T>(value));
  return true;
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, T &&value) {
  auto *const v = GetOrCreateAnyValue(name);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(std::forward<T>(value));
  return true;
}
template<typename T>
bool AttrStore::SetByName(const std::string &name, const T &value) {
  auto *const v = GetOrCreateAnyValue(name);
  if (v == nullptr) {
    return false;
  }
  (void)v->SetValue(value);
  return true;
}

template<typename T>
const T *AttrStore::Get(const AttrId attr_id) const {
  auto *const v = GetAnyValue(attr_id);
  if (v == nullptr) {
    return nullptr;
  }
  return v->Get<T>();
}
template<typename T>
const T *AttrStore::GetByName(const std::string &name) const {
  auto *const v = GetAnyValue(name);
  if (v == nullptr) {
    return nullptr;
  }
  return v->Get<T>();
}

template<typename T>
T *AttrStore::MutableGet(const AttrId attr_id) {
  auto *const v = MutableAnyValue(attr_id);
  if (v == nullptr) {
    return nullptr;
  }
  return v->MutableGet<T>();
}
template<typename T>
T *AttrStore::MutableGetByName(const std::string &name) {
  auto *const v = MutableAnyValue(name);
  if (v == nullptr) {
    return nullptr;
  }
  return v->MutableGet<T>();
}

}  // namespace ge

#endif  // EXECUTE_GRAPH_ATTR_STORE_H
