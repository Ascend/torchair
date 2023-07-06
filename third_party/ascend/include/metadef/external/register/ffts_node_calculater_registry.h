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

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_FFTS_NODE_CALCULATER_REGISTRY_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_FFTS_NODE_CALCULATER_REGISTRY_H_
#include <string>
#include <functional>
#include <vector>
#include "graph/node.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
struct NodeMemPara {
    size_t size {0};
    void *dev_addr {nullptr};
    void *host_addr {nullptr};
};

class FFTSNodeCalculaterRegistry {
 public:
  /*
   * Output param:
   *   1.size_t total_size -- node need memory size
   *   2.size_t pre_data_size -- node pre proc data size
   *   3.std::unique_ptr<uint8_t[]> pre_data_ptr -- node pre proc data memory with size(pre_data_size), framework will
   *   copy pre_data to new alloc memory
   */
  using NodeCalculater = ge::graphStatus (*)(const ge::NodePtr &node, const LoweringGlobalData *global_data,
      size_t &total_size, size_t &pre_data_size, std::unique_ptr<uint8_t[]> &pre_data_ptr);
  static FFTSNodeCalculaterRegistry &GetInstance();
  NodeCalculater FindNodeCalculater(const std::string &func_name);
  void Register(const std::string &func_name, const NodeCalculater func);

 private:
  std::unordered_map<std::string, NodeCalculater> names_to_calculater_;
};

class FFTSNodeCalculaterRegister {
 public:
    FFTSNodeCalculaterRegister(const string &func_name, FFTSNodeCalculaterRegistry::NodeCalculater func) noexcept;
};
}  // namespace gert

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define GERT_REGISTER_FFTS_NODE_CALCULATER_COUNTER2(type, func, counter)                  \
  static const gert::FFTSNodeCalculaterRegister g_register_node_calculater_##counter ATTRIBUTE_USED =  \
      gert::FFTSNodeCalculaterRegister(type, func)
#define GERT_REGISTER_FFTS_NODE_CALCULATER_COUNTER(type, func, counter)                    \
  GERT_REGISTER_FFTS_NODE_CALCULATER_COUNTER2(type, func, counter)
#define FFTS_REGISTER_NODE_CALCULATER(type, func)                                \
  GERT_REGISTER_FFTS_NODE_CALCULATER_COUNTER(type, func, __COUNTER__)

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_FFTS_NODE_CALCULATER_REGISTRY_H_
