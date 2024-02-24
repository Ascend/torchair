/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef INC_OP_IMPL_SPACE_REGISTRY_H_
#define INC_OP_IMPL_SPACE_REGISTRY_H_

#include "register/op_impl_registry_holder_manager.h"

namespace gert {
using OpTypesToImplMap = std::map<OpImplKernelRegistry::OpType, OpImplKernelRegistry::OpImplFunctions>;
class OpImplSpaceRegistry : public std::enable_shared_from_this<OpImplSpaceRegistry> {
 public:
  OpImplSpaceRegistry() = default;

  ~OpImplSpaceRegistry() = default;

  ge::graphStatus GetOrCreateRegistry(const std::vector<ge::OpSoBinPtr> &bins, const ge::SoInOmInfo &so_info);

  ge::graphStatus AddRegistry(const std::shared_ptr<OpImplRegistryHolder> &registry_holder);

  const OpImplKernelRegistry::OpImplFunctions *GetOpImpl(const std::string &op_type) const;

  const OpImplKernelRegistry::PrivateAttrList &GetPrivateAttrs(const std::string &op_type) const;

  static ge::graphStatus LoadSoAndSaveToRegistry(const std::string &so_path);

 private:
  void MergeTypesToImpl(OpTypesToImplMap &merged_impl, OpTypesToImplMap &src_impl) const;

  void MergeFunctions(OpImplKernelRegistry::OpImplFunctions &merged_funcs,
                      const OpImplKernelRegistry::OpImplFunctions &src_funcs, const std::string &op_type) const;
 private:
  std::vector<std::shared_ptr<OpImplRegistryHolder>> op_impl_registries_;
  OpTypesToImplMap merged_types_to_impl_;
};
using OpImplSpaceRegistryPtr = std::shared_ptr<OpImplSpaceRegistry>;

class DefaultOpImplSpaceRegistry {
 public:
  DefaultOpImplSpaceRegistry() = default;

  ~DefaultOpImplSpaceRegistry() = default;

  static DefaultOpImplSpaceRegistry &GetInstance();

  OpImplSpaceRegistryPtr &GetDefaultSpaceRegistry() {
    return space_registry_;
  }

  void SetDefaultSpaceRegistry(const OpImplSpaceRegistryPtr &space_registry) {
    space_registry_ = space_registry;
  }

 private:
  OpImplSpaceRegistryPtr space_registry_ = nullptr;
};
}  // namespace gert
#endif  // INC_OP_IMPL_SPACE_REGISTRY_H_
