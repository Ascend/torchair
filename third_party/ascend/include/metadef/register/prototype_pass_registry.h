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

#ifndef METADEF_PROTOTYPE_PASS_REGISTRY_H
#define METADEF_PROTOTYPE_PASS_REGISTRY_H

#include <google/protobuf/message.h>

#include <functional>
#include <vector>
#include <memory>

#include "external/ge_common/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "register/register_error_codes.h"
#include "register/register_fmk_types.h"

namespace ge {
class ProtoTypeBasePass {
 public:
  ProtoTypeBasePass() = default;
  virtual Status Run(google::protobuf::Message *message) = 0;
  virtual ~ProtoTypeBasePass() = default;

 private:
  ProtoTypeBasePass(const ProtoTypeBasePass &) = delete;
  ProtoTypeBasePass &operator=(const ProtoTypeBasePass &) & = delete;
};

class ProtoTypePassRegistry {
 public:
  using CreateFn = std::function<ProtoTypeBasePass *(void)>;
  ~ProtoTypePassRegistry();

  static ProtoTypePassRegistry &GetInstance();

  void RegisterProtoTypePass(const char_t *const pass_name, const CreateFn &create_fn,
                             const domi::FrameworkType fmk_type);

  std::vector<std::pair<std::string, CreateFn>> GetCreateFnByType(const domi::FrameworkType fmk_type) const;

 private:
  ProtoTypePassRegistry();
  class ProtoTypePassRegistryImpl;
  std::unique_ptr<ProtoTypePassRegistryImpl> impl_;
};

class ProtoTypePassRegistrar {
 public:
  ProtoTypePassRegistrar(const char_t *const pass_name, ProtoTypeBasePass *(*const create_fn)(),
                         const domi::FrameworkType fmk_type);
  ~ProtoTypePassRegistrar() = default;
};
}  // namespace ge

#define REGISTER_PROTOTYPE_PASS(pass_name, pass, fmk_type) \
  REGISTER_PROTOTYPE_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass, fmk_type)

#define REGISTER_PROTOTYPE_PASS_UNIQ_HELPER(ctr, pass_name, pass, fmk_type) \
  REGISTER_PROTOTYPE_PASS_UNIQ(ctr, pass_name, pass, fmk_type)

#define REGISTER_PROTOTYPE_PASS_UNIQ(ctr, pass_name, pass, fmk_type)                         \
  static ::ge::ProtoTypePassRegistrar register_prototype_pass##ctr __attribute__((unused)) = \
      ::ge::ProtoTypePassRegistrar(                                                          \
          (pass_name), []()->::ge::ProtoTypeBasePass * { return new (std::nothrow) pass(); }, (fmk_type))

#endif  // METADEF_PROTOTYPE_PASS_REGISTRY_H
