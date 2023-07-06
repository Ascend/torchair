/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
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

#ifndef GRAPH_CACHE_POLICY_CACHE_POLICY_H_
#define GRAPH_CACHE_POLICY_CACHE_POLICY_H_

#include <vector>
#include <memory>
#include "cache_state.h"
#include "policy_register.h"
#include "graph/ge_error_codes.h"

namespace ge {
class CachePolicy {
 public:
  ~CachePolicy() = default;

  CachePolicy(const CachePolicy &) = delete;
  CachePolicy(CachePolicy &&) = delete;
  CachePolicy &operator=(const CachePolicy &) = delete;
  CachePolicy &operator=(CachePolicy &&) = delete;

  static std::unique_ptr<CachePolicy> Create(const MatchPolicyPtr &mp, const AgingPolicyPtr &ap);
  static std::unique_ptr<CachePolicy> Create(const MatchPolicyType mp_type, const AgingPolicyType ap_type,
                                             size_t cached_aging_depth = kDefaultCacheQueueDepth);

  graphStatus SetMatchPolicy(const MatchPolicyPtr mp);

  graphStatus SetAgingPolicy(const AgingPolicyPtr ap);

  CacheItemId AddCache(const CacheDescPtr &cache_desc);

  CacheItemId FindCache(const CacheDescPtr &cache_desc) const;

  std::vector<CacheItemId> DeleteCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DeleteCache(const std::vector<CacheItemId> &delete_item);

  std::vector<CacheItemId> DoAging();

  CachePolicy() = default;

 private:
  CacheState compile_cache_state_;
  MatchPolicyPtr mp_ = nullptr;
  AgingPolicyPtr ap_ = nullptr;
};
}  // namespace ge
#endif