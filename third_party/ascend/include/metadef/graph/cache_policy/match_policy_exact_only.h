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
#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_MATCH_POLICY_EXACT_ONLY_H_
#include "graph/cache_policy/match_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class MatchPolicyExactOnly : public MatchPolicy {
public:
  CacheItemId GetCacheItemId(const CCStatType &cc_state, const CacheDescPtr &desc) const override;
  ~MatchPolicyExactOnly() override = default;
};

REGISTER_MATCH_POLICY_CREATOR(MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
                              []() {
                                return make_shared<MatchPolicyExactOnly>();
                              });
}  // namespace ge
#endif

