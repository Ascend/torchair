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

#ifndef GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_POLICY_REGISTER_H_
#define GRAPH_CACHE_POLICY_POLICY_MANAGEMENT_POLICY_REGISTER_H_
#include <map>
#include <mutex>
#include "common/checker.h"
#include "match_policy.h"
#include "aging_policy.h"

namespace ge {
using MatchPolicyPtr = std::shared_ptr<MatchPolicy>;
using AgingPolicyPtr = std::shared_ptr<AgingPolicy>;
using MatchPolicyCreator = std::function<MatchPolicyPtr()>;
using AgingPolicyCreator = std::function<AgingPolicyPtr()>;
enum class MatchPolicyType {
  MATCH_POLICY_EXACT_ONLY = 0,
  MATCH_POLICY_FOR_EXACTLY_THE_SAME = 1
};
enum class AgingPolicyType {
  AGING_POLICY_LRU = 0,
  AGING_POLICY_LRU_K = 1
};

class PolicyRegister {
 public:
  ~PolicyRegister() = default;
  PolicyRegister(const PolicyRegister&) = delete;
  PolicyRegister &operator=(const PolicyRegister &other) = delete;
  static PolicyRegister &GetInstance();
  void RegisterMatchPolicy(const MatchPolicyType match_policy_type, const MatchPolicyCreator &creator) {
    const std::lock_guard<std::mutex> lock(mu_);
    (void)match_policy_registry_.emplace(match_policy_type, creator);
    return;
  }

  void RegisterAgingPolicy(const AgingPolicyType aging_policy_type, const AgingPolicyCreator &creator) {
    const std::lock_guard<std::mutex> lock(mu_);
    (void)aging_policy_registry_.emplace(aging_policy_type, creator);
  }

  MatchPolicyPtr GetMatchPolicy(const MatchPolicyType match_policy_type) {
    const auto iter = match_policy_registry_.find(match_policy_type);
    if (iter != match_policy_registry_.end()) {
      GE_ASSERT_NOTNULL(iter->second, "[GetMatchPolicy] failed. Match policy type : %d was incorrectly registered",
                        static_cast<int32_t>(match_policy_type));
      return iter->second();
    }
    GELOGE(ge::GRAPH_FAILED, "[GetMatchPolicy] failed. Match policy type : %d  has not been registered",
           static_cast<int32_t>(match_policy_type));
    return nullptr;
  }
  AgingPolicyPtr GetAgingPolicy(const AgingPolicyType aging_policy_type) {
    const auto iter = aging_policy_registry_.find(aging_policy_type);
    if (iter != aging_policy_registry_.end()) {
      GE_ASSERT_NOTNULL(iter->second, "[GetAgingPolicy] failed. Aging policy type : %d was incorrectly registered",
                        static_cast<int32_t>(aging_policy_type));
      return iter->second();
    }
    GELOGE(ge::GRAPH_FAILED, "[GetAgingPolicy] failed. Aging policy type : %d  has not been registered",
           static_cast<int32_t>(aging_policy_type));
    return nullptr;
  }
 private:
  PolicyRegister() = default;
  std::mutex mu_;
  std::map<MatchPolicyType, MatchPolicyCreator> match_policy_registry_;
  std::map<AgingPolicyType, AgingPolicyCreator> aging_policy_registry_;
};

class MatchPolicyRegister {
 public:
  MatchPolicyRegister(const MatchPolicyType match_policy_type, const MatchPolicyCreator &creator);
  ~MatchPolicyRegister() = default;
};

class AgingPolicyRegister {
 public:
  AgingPolicyRegister(const AgingPolicyType aging_policy_type, const AgingPolicyCreator &creator);
  ~AgingPolicyRegister() = default;
};

#define REGISTER_MATCH_POLICY_CREATOR_COUNTER(policy_type, func, counter)                                              \
  static MatchPolicyRegister match_policy_register##counter(policy_type, func)
#define REGISTER_MATCH_POLICY_CREATOR_COUNTER_NUMBER(policy_type, func, counter)                                       \
  REGISTER_MATCH_POLICY_CREATOR_COUNTER(policy_type, func, counter)
#define REGISTER_MATCH_POLICY_CREATOR(policy_type, func)                                                               \
  REGISTER_MATCH_POLICY_CREATOR_COUNTER_NUMBER(policy_type, func, __COUNTER__)

#define REGISTER_AGING_POLICY_CREATOR_COUNTER(policy_type, func, counter)                                              \
  static AgingPolicyRegister aging_policy_register##counter(policy_type, func)
#define REGISTER_AGING_POLICY_CREATOR_COUNTER_NUMBER(policy_type, func, counter)                                       \
  REGISTER_AGING_POLICY_CREATOR_COUNTER(policy_type, func, counter)
#define REGISTER_AGING_POLICY_CREATOR(policy_type, func)                                                               \
  REGISTER_AGING_POLICY_CREATOR_COUNTER_NUMBER(policy_type, func, __COUNTER__)
} // namespace ge
#endif