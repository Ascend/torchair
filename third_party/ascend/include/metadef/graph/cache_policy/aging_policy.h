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

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_H_
#include "graph/cache_policy/cache_state.h"

namespace ge {
constexpr const size_t kDefaultCacheQueueDepth = 1U;
class AgingPolicy {
 public:
  AgingPolicy() = default;
  virtual ~AgingPolicy() = default;
  virtual void SetCachedAgingDepth(size_t depth) = 0;
  virtual std::vector<CacheItemId> DoAging(const CacheState &cache_state) const = 0;
  virtual bool IsReadyToAddCache(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) = 0;
 private:
  AgingPolicy &operator=(const AgingPolicy &anging_polocy) = delete;
  AgingPolicy(const AgingPolicy &anging_polocy) = delete;
};
}
#endif