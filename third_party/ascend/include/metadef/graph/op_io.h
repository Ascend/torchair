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

#ifndef METADEF_CXX_OP_IO_H
#define METADEF_CXX_OP_IO_H
namespace ge {

class OpIO {
 public:
  OpIO(const std::string &name, const int32_t index, const OperatorImplPtr &owner)
      : name_(name), index_(index), owner_(owner) {}

  ~OpIO() = default;

  std::string GetName() const { return name_; }

  int32_t GetIndex() const { return index_; }

  OperatorImplPtr GetOwner() const { return owner_; }

 private:
  std::string name_;
  int32_t index_;
  std::shared_ptr<OperatorImpl> owner_;
};
}
#endif  // METADEF_CXX_OP_IO_H
