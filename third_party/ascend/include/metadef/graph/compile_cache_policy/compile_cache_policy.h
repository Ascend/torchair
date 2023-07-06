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

#ifndef GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_POLICY_H_
#define GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_POLICY_H_

#include "graph/cache_policy/cache_policy.h"

#include <vector>
#include <memory>
namespace ge {
class CompileCachePolicy {
public:
  ~CompileCachePolicy() = default;

  CompileCachePolicy(const CompileCachePolicy &) = delete;
  CompileCachePolicy(CompileCachePolicy &&) = delete;
  CompileCachePolicy &operator=(const CompileCachePolicy &) = delete;
  CompileCachePolicy &operator=(CompileCachePolicy &&) = delete;

  static std::unique_ptr<CompileCachePolicy> Create(const MatchPolicyPtr mp, const AgingPolicyPtr ap);
  static std::unique_ptr<CompileCachePolicy> Create(const MatchPolicyType mp_type, const AgingPolicyType ap_type);

  graphStatus SetMatchPolicy(const MatchPolicyPtr mp);

  graphStatus SetAgingPolicy(const AgingPolicyPtr ap);

  CacheItemId AddCache(const CompileCacheDesc &compile_cache_desc);

  CacheItemId FindCache(const CompileCacheDesc &compile_cache_desc) const;

  std::vector<CacheItemId> DeleteCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DeleteCache(const std::vector<CacheItemId> &delete_item);

  std::vector<CacheItemId> DoAging();

  CompileCachePolicy() = default;

private:
  CacheState compile_cache_state_;
  MatchPolicyPtr mp_ = nullptr;
  AgingPolicyPtr ap_ = nullptr;
};
}  // namespace ge
#endif