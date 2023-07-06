/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "register/graph_optimizer/fusion_common/fusion_pass_desc.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"

namespace fe {
class FusionPassRegistry {
 public:
  using CreateFn = GraphPass *(*)();
  struct PassDesc {
    PassAttr attr;
    CreateFn create_fn;
  };
  ~FusionPassRegistry();

  static FusionPassRegistry &GetInstance();

  void RegisterPass(const GraphFusionPassType &pass_type, const std::string &pass_name, CreateFn create_fn,
                    PassAttr attr) const;

  std::map<std::string, PassDesc> GetPassDesc(const GraphFusionPassType &pass_type);

  std::map<std::string, CreateFn> GetCreateFnByType(const GraphFusionPassType &pass_type);

 private:
  FusionPassRegistry();
  class FusionPassRegistryImpl;
  std::unique_ptr<FusionPassRegistryImpl> impl_;
};

class FusionPassRegistrar {
 public:
  FusionPassRegistrar(const GraphFusionPassType &pass_type, const std::string &pass_name,
                      GraphPass *(*create_fn)(), PassAttr attr);

  ~FusionPassRegistrar() {}
};

#define REGISTER_PASS(pass_name, pass_type, pass_class) \
  REG_PASS(pass_name, pass_type, pass_class, 0)

#define REG_PASS(pass_name, pass_type, pass_class, attr) \
  REG_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, pass_class, attr)

#define REG_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, pass_class, attr) \
  REG_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)

#define REG_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)                                                 \
  static ::fe::FusionPassRegistrar register_fusion_pass##ctr __attribute__((unused)) = ::fe::FusionPassRegistrar( \
      pass_type, pass_name, []() -> ::fe::GraphPass * { return new (std::nothrow) pass_class(); }, attr)
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_
