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

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_LRU_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_AGING_POLICY_LRU_H_
#include "graph/cache_policy/aging_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class AgingPolicyLru : public AgingPolicy {
public:
  virtual ~AgingPolicyLru() override = default;
  void SetDeleteInterval(const uint64_t &interval) {
    delete_interval_ = interval;
  }
  void SetCachedAgingDepth(size_t depth) override {
    (void)depth;
  }
  bool IsReadyToAddCache(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) override {
    (void) hash_key;
    (void) cache_desc;
    return true;
  }
  std::vector<CacheItemId> DoAging(const CacheState &cache_state) const override;

private:
  uint64_t delete_interval_ = 0U;
};

REGISTER_AGING_POLICY_CREATOR(AgingPolicyType::AGING_POLICY_LRU,
                              []() {
                                return make_shared<AgingPolicyLru>();
                              });
}  // namespace ge
#endif