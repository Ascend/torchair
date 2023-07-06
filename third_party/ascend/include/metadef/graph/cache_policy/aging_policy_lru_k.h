/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef METADEF_CXX_GRAPH_CACHE_POLICY_AGING_POLICY_LRU_K_H
#define METADEF_CXX_GRAPH_CACHE_POLICY_AGING_POLICY_LRU_K_H
#include "graph/cache_policy/aging_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class AgingPolicyLruK : public AgingPolicy {
 public:
  AgingPolicyLruK() : depth_(kDefaultCacheQueueDepth) {}
  explicit AgingPolicyLruK(size_t depth) : depth_(depth) {}
  AgingPolicyLruK(size_t k_times, size_t depth) : k_times_(k_times), depth_(depth) {}
  ~AgingPolicyLruK() override = default;

  void SetCachedAgingDepth(size_t depth) override {
    depth_ = depth;
  }
  bool IsReadyToAddCache(const CacheHashKey hash_key, const CacheDescPtr &cache_desc) override {
    return IsCacheDescAppearKTimes(hash_key, cache_desc);
  }
  std::vector<CacheItemId> DoAging(const CacheState &cache_state) const override;

 private:
  bool IsCacheDescAppearKTimes(const CacheHashKey hash_key, const CacheDescPtr &cache_desc);
 private:
  size_t k_times_ = 2U;
  size_t depth_;
  // todo 历史缓存队列的老化
  std::mutex hash_2_cache_descs_and_count_mu_;
  std::unordered_map<CacheHashKey, std::vector<std::pair<const CacheDescPtr, size_t>>> hash_2_cache_descs_and_count_;
};
REGISTER_AGING_POLICY_CREATOR(AgingPolicyType::AGING_POLICY_LRU_K,
                              []() { return std::make_shared<AgingPolicyLruK>(); });
}
#endif  // METADEF_CXX_GRAPH_CACHE_POLICY_AGING_POLICY_LRU_K_H
