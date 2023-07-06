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

#ifndef INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_
#define INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_

#include <map>
#include <memory>
#include <string>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include "graph/detail/any_map.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/attr_store.h"

namespace google {
namespace protobuf {
class Message;
template<typename Key, typename T>
class Map;
}  // namespace protobuf
}  // namespace google

namespace ge {
namespace proto {
class AttrDef;
class TensorDef;
class TensorDescriptor;
class ShapeDef;
class NamedAttrs;
class ModelDef;
class OpDef;
class GraphDef;
}  // namespace proto

using ProtoAttrMap = AttrStore;
using ConstProtoAttrMap = const AttrStore;
using ProtoMsgOwner = std::shared_ptr<::google::protobuf::Message>;

template<class ProtoType>
class GeIrProtoHelper {
 public:
  GeIrProtoHelper(const ProtoMsgOwner &protoOwner, ProtoType *const protoMsg)
      : protoOwner_(protoOwner), protoMsg_(protoMsg) {}

  GeIrProtoHelper() {
    protoOwner_ = std::shared_ptr<::google::protobuf::Message>(nullptr);
    protoMsg_ = nullptr;
  }
  virtual ~GeIrProtoHelper() = default;

  template<typename T>
  GeIrProtoHelper(const GeIrProtoHelper<T> &other) {
    protoOwner_ = other.protoOwner_;
    protoMsg_ = other.protoMsg_;
  }

  GeIrProtoHelper(const GeIrProtoHelper<ProtoType> &other) {
    protoOwner_ = other.protoOwner_;
    protoMsg_ = other.protoMsg_;
  }
  template<typename T>
  GeIrProtoHelper &operator=(const GeIrProtoHelper<T> &other) {
    protoOwner_ = other.protoOnwer_;
    protoMsg_ = other.protoMsg_;
    return *this;
  }

  GeIrProtoHelper &operator=(const GeIrProtoHelper<ProtoType> &other) {
    if (this != &other) {
      protoOwner_ = other.protoOwner_;
      protoMsg_ = other.protoMsg_;
    }
    return *this;
  }
  void InitDefault();
  template<typename T>
  bool operator==(const GeIrProtoHelper<T> &other) const {
    return (protoOwner_ == other.protoOwner_) && (protoMsg_ == other.protoMsg_);
  }

  inline const ProtoMsgOwner &GetProtoOwner() const {
    return protoOwner_;
  }
  inline ProtoType *GetProtoMsg() const {
    return protoMsg_;
  }
  void CopyValueFrom(const GeIrProtoHelper<const ProtoType> &other) {
    if ((other.protoMsg_ != nullptr) && (protoMsg_ != nullptr)) {
      *protoMsg_ = *other.protoMsg_;
    }
  }
  void MoveValueFrom(GeIrProtoHelper<ProtoType> &&other) {
    if ((other.protoMsg_ != nullptr) && (protoMsg_ != nullptr)) {
      *protoMsg_ = std::move(*other.protoMsg_);
    }
  }

  void Swap(GeIrProtoHelper<ProtoType> &other) {
    protoOwner_.swap(other.protoOwner_);

    ProtoType *const temp = protoMsg_;
    protoMsg_ = other.protoMsg_;
    other.protoMsg_ = temp;
  }

  friend class GeIrProtoHelper<typename std::conditional<
      std::is_const<ProtoType>::value, typename std::remove_const<ProtoType>::type, const ProtoType>::type>;
  friend class ComputerGraphImpl;
  friend class GeTensorSerializeUtils;

 private:
  // protoMsg_ is part of protoOwner_, they have the same runtime
  ProtoMsgOwner protoOwner_ = nullptr;
  ProtoType *protoMsg_ = nullptr;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrHolder {
 public:
  AttrHolder() = default;
  virtual ~AttrHolder() = default;
  /**
   * 对当前AttrHolder对象设置属性，属性名为`name`,属性值为`value`,
   * 需要注意的是 如果当前对象已经存在了`name`类型的属性，接口会进行刷新值的操作
   * @param name
   * @param value
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  graphStatus SetAttr(const std::string &name, const AnyValue &value);
  /**
   * 尝试对当前AttrHolder对象设置属性，属性名为`name`,属性值为`value`,
   * 需要注意的是 如果当前对象已经存在了`name`类型的属性，接口并不会进行刷新值的操作
   * @param name
   * @param value
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  graphStatus TrySetAttr(const std::string &name, const AnyValue &value);

  graphStatus GetAttr(const std::string &name, AnyValue &value) const;

  bool HasAttr(const std::string &name) const;

  graphStatus DelAttr(const std::string &name);

  void CopyAttrsFrom(const AttrHolder &holder);

  void CopyFrom(const AttrHolder &holder);

  void SwapBase(AttrHolder &holder) {
    required_attrs_.swap(holder.required_attrs_);
    ext_attrs_.Swap(holder.ext_attrs_);
  }
  /**
   * 对当前对象设置名称为name, 值为value，类型为T的属性，如果对象已经存在了
   * 名称name和类型T的属性，那么此接口会刷新属性的值，需要注意的是如果对象已
   * 经存在了名称name和非类型T的属性，设置行为会失败告终
   * @param name 属性的名称
   * @param value 任意类型的属性值
   * @return true/false 设置成功返回true, 设置失败返回false
   */
  template<class T>
  bool SetExtAttr(const std::string &name, const T &value) {
    return ext_attrs_.Set(name, value);
  }
  /**
   * 对当前对象尝试获取名称name, 类型为T的属性值，如果对象没有name名称的属性，或者
   * 属性的类型不为T, 查询行为失败
   * @param name 属性的名称
   * @param defaultValue 默认值，用于查询失败时返回这个默认值
   * @return 如果查询成功，返回查询到的属性值，如果查询失败，返回传入的默认值
   */
  template<class T>
  T TryGetExtAttr(const std::string &name, const T defaultValue) const {
    T ret(defaultValue);
    (void) ext_attrs_.Get(name, ret);
    return ret;
  }

  /**
   * 对当前对象尝试获取名称name, 类型为T的属性值，如果对象没有name名称的属性，或者
   * 属性的类型不为T, 查询行为失败
   * @param name 属性的名称
   * @return 如果查询成功，返回查询到的属性值的指针，如果查询失败，返回空指针
   */
  template<class T>
  const T *GetExtAttr(const std::string &name) const {
    return ext_attrs_.Get<T>(name);
  }

  template<class T>
  T *GetExtAttr(const std::string &name) {
    return const_cast<T *>(ext_attrs_.Get<T>(name));
  }

  bool DelExtAttr(const std::string &name) {
    return ext_attrs_.Erase(name);
  }

 protected:
  graphStatus AddRequiredAttr(const std::string &name);
  const std::set<std::string> GetAllAttrNames() const;
  const std::map<std::string, AnyValue> GetAllAttrs() const;

  virtual ProtoAttrMap &MutableAttrMap() = 0;
  virtual ConstProtoAttrMap &GetAttrMap() const = 0;

  friend class ModelSerializeImp;
  friend class AttrUtils;
  friend class OpDescUtils;
  friend class GraphUtils;
  std::vector<std::string> required_attrs_;

 private:
  AnyMap ext_attrs_;
};
}  // namespace ge
#endif  // INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_
