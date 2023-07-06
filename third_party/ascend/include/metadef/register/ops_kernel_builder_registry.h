/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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

#ifndef INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H
#define INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H

#include <memory>
#include "register/register_types.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
using OpsKernelBuilderPtr = std::shared_ptr<OpsKernelBuilder>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistry {
 public:
  ~OpsKernelBuilderRegistry() noexcept;
  static OpsKernelBuilderRegistry &GetInstance();

  void Register(const std::string &lib_name, const OpsKernelBuilderPtr &instance);

  void Unregister(const std::string &lib_name);

  void UnregisterAll();

  const std::map<std::string, OpsKernelBuilderPtr> &GetAll() const;

 private:
  OpsKernelBuilderRegistry() = default;
  std::map<std::string, OpsKernelBuilderPtr> kernel_builders_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistrar {
 public:
  using CreateFn = OpsKernelBuilder *(*)();
  OpsKernelBuilderRegistrar(const std::string &kernel_lib_name, const CreateFn fn);
  ~OpsKernelBuilderRegistrar() noexcept;

private:
  std::string kernel_lib_name_;
};
}  // namespace ge

#define REGISTER_OPS_KERNEL_BUILDER(kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)                         \
  static ::ge::OpsKernelBuilderRegistrar register_op_kernel_builder_##ctr                       \
      __attribute__((unused)) =                                                                 \
          ::ge::OpsKernelBuilderRegistrar((kernel_lib_name), []()->::ge::OpsKernelBuilder* {    \
            return new (std::nothrow) (builder)();                                              \
          })

#endif // INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H
