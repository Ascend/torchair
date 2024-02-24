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

#ifndef INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_
#define INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_

#include <string>
#include <unordered_map>
#include <map>
#include <mutex>
#include "graph/op_so_bin.h"
#include "register/op_impl_registry_api.h"

namespace gert {
class OpImplRegistryHolder {
 public:
  OpImplRegistryHolder() = default;

  virtual ~OpImplRegistryHolder();

  std::map<OpImplKernelRegistry::OpType, OpImplKernelRegistry::OpImplFunctions> &GetTypesToImpl() {
    return types_to_impl_;
  }

  void SetHandle(const void *handle) { handle_ = const_cast<void *>(handle); }

  std::unique_ptr<TypesToImpl[]> GetOpImplFunctionsByHandle(const void *handle,
                                                            const std::string &so_path,
                                                            size_t &impl_num) const;

  void AddTypesToImpl(const gert::OpImplKernelRegistry::OpType op_type,
                      const gert::OpImplKernelRegistry::OpImplFunctions funcs);

 protected:
  std::map<OpImplKernelRegistry::OpType, OpImplKernelRegistry::OpImplFunctions> types_to_impl_;
  void *handle_ = nullptr;
};
using OpImplRegistryHolderPtr = std::shared_ptr<OpImplRegistryHolder>;

class OmOpImplRegistryHolder : public OpImplRegistryHolder {
 public:
  OmOpImplRegistryHolder() = default;

  virtual ~OmOpImplRegistryHolder() = default;

  ge::graphStatus LoadSo(const std::shared_ptr<ge::OpSoBin> &so_bin);

 private:
  ge::graphStatus CreateOmOppDir(std::string &opp_dir) const;

  ge::graphStatus RmOmOppDir(const std::string &opp_dir) const;

  ge::graphStatus SaveToFile(const std::shared_ptr<ge::OpSoBin> &so_bin, const std::string &opp_path) const;
};

class OpImplRegistryHolderManager {
 public:
  OpImplRegistryHolderManager() = default;

  ~OpImplRegistryHolderManager();

  static OpImplRegistryHolderManager &GetInstance();

  void AddRegistry(std::string &so_data, const std::shared_ptr<OpImplRegistryHolder> &registry_holder);

  void UpdateOpImplRegistries();

  const OpImplRegistryHolderPtr GetOpImplRegistryHolder(std::string &so_data);

  OpImplRegistryHolderPtr GetOrCreateOpImplRegistryHolder(std::string &so_data,
                                                          const std::string &so_name,
                                                          const ge::SoInOmInfo &so_info,
                                                          const std::function<OpImplRegistryHolderPtr()> create_func);

  size_t GetOpImplRegistrySize() {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    return op_impl_registries_.size();
  }

  void ClearOpImplRegistries() {
    const std::lock_guard<std::mutex> lock(map_mutex_);
    op_impl_registries_.clear();
  }

 private:
  /**
   * 背景：当前加载的so里边包含了算子原型等自注册(如OperatorFactoryImpl::operator_infer_axis_type_info_funcs等static变量，在进程退出前析构)。
   * 原方案使用weak_ptr管理OpImplRegistryHolder，不影响生命周期，引用计数一旦减为0，将触发析构，并关闭so的句柄；
   * 如果OpImplRegistryHolder在比较早的时机析构，那么进程退出时，operator_infer_axis_type_info_funcs这些static变量将无法正常析构。
   * 此处临时改为shared_ptr，使OpImplRegistryHolder与本类单例的生命周期一致，从而能够确保在进程退出前才触发析构，规避上述问题。
   * todo 此处是规避方案，后续将继续使用weak_ptr来管理OpImplRegistryHolder，后续将梳理上述自注册机制，修改成space_registry注册机制
   * */
  std::unordered_map<std::string, std::shared_ptr<OpImplRegistryHolder>> op_impl_registries_;
  std::mutex map_mutex_;
};
}  // namespace gert
#endif  // INC_OP_IMPL_REGISTRY_HOLDER_MANAGER_H_
