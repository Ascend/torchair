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

#ifndef INC_GRAPH_DETAIL_ANY_MAP_H_
#define INC_GRAPH_DETAIL_ANY_MAP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <set>

#include "graph/compiler_options.h"

namespace ge {
class TypeID {
 public:
  template <class T>
  static TypeID Of() {
    return TypeID(METADEF_FUNCTION_IDENTIFIER);
  }

  ~TypeID() = default;

  bool operator==(const TypeID &other) const { return type_ == other.type_; }
  bool operator!=(const TypeID &other) const { return type_ != other.type_; }

 private:
  explicit TypeID(const std::string &type) : type_(type) {}

  std::string type_;
};

class AnyMap {
 public:
  template <class DT>
  bool Set(const std::string &name, const DT &val);

  template <class T>
  bool Get(const std::string &name, T &ret_value) const;

  template<class T>
  const T *Get(const std::string &name) const;

  bool Has(const std::string &name) const { return any_values_.find(name) != any_values_.end(); }

  void Swap(AnyMap &other) {
    any_values_.swap(other.any_values_);
  }

  void Names(std::set<std::string> &names) const {
    for (const auto &item : any_values_) {
      (void)names.emplace(item.first);
    }
  }
  bool Erase(const std::string &name) { return (any_values_.erase(name) > 0);}

 private:
  class Placeholder {
   public:
    Placeholder() = default;
    virtual ~Placeholder() = default;
    Placeholder(const Placeholder &) = delete;
    Placeholder &operator=(const Placeholder &) = delete;
    Placeholder(Placeholder &&) = delete;
    Placeholder &operator=(Placeholder &&) = delete;
    virtual const TypeID &GetTypeInfo() const = 0;
  };

  template <typename VT>
  class Holder : public Placeholder {
   public:
    explicit Holder(const VT &value) : Placeholder(), value_(value) {}

    ~Holder() override = default;

    const TypeID &GetTypeInfo() const override {
      static const TypeID typeId = TypeID::Of<VT>();
      return typeId;
    }

    friend class AnyMap;

   private:
    const VT value_;
  };

  std::map<std::string, std::shared_ptr<Placeholder>> any_values_;
};

template <class DT>
bool AnyMap::Set(const std::string &name, const DT &val) {
  const auto it = any_values_.find(name);

  std::shared_ptr<Holder<DT>> tmp;
  try {
    tmp = std::make_shared<Holder<DT>>(val);
  } catch (...) {
    tmp = nullptr;
    return false;
  }

  if (it == any_values_.end()) {
    (void) any_values_.emplace(name, tmp);
  } else {
    if (it->second && (it->second->GetTypeInfo() == TypeID::Of<DT>())) {
      it->second = tmp;
    } else {
      return false;
    }
  }
  return true;
}

template <class T>
bool AnyMap::Get(const std::string &name, T &ret_value) const {
  auto tp = Get<T>(name);
  if (tp == nullptr) {
    return false;
  }
  ret_value = *tp;
  return true;
}

template <class T>
const T *AnyMap::Get(const std::string &name) const {
  const auto iter = any_values_.find(name);
  if (iter == any_values_.end()) {
    return nullptr;
  }
  if (iter->second->GetTypeInfo() != TypeID::Of<T>()) {
    return nullptr;
  }
  const auto holder = static_cast<Holder<T>*>(iter->second.get());
  return &holder->value_;
}
}  // namespace ge
#endif  // INC_GRAPH_DETAIL_ANY_MAP_H_
