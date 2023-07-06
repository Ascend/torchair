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

#ifndef METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
#define METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
#include "graph/cache_policy/match_policy.h"
#include "graph/cache_policy/policy_register.h"

namespace ge {
class MatchPolicyForExactlyTheSame : public MatchPolicy {
 public:
  MatchPolicyForExactlyTheSame() = default;
  ~MatchPolicyForExactlyTheSame() override = default;

  CacheItemId GetCacheItemId(const CCStatType &cc_state, const CacheDescPtr &cache_desc) const override;
};
REGISTER_MATCH_POLICY_CREATOR(MatchPolicyType::MATCH_POLICY_FOR_EXACTLY_THE_SAME,
                              []() { return std::make_shared<MatchPolicyForExactlyTheSame>(); });
}  // namespace ge
#endif  // METADEF_CXX_GRAPH_CACHE_POLICY_MATCH_POLICY_FOR_EXACTLY_THE_SAME_H
