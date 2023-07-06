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

#ifndef GRAPH_CACHE_POLICY_CACHE_STATE_H
#define GRAPH_CACHE_POLICY_CACHE_STATE_H

#include <vector>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>

#include "compile_cache_desc.h"

namespace ge {
class CacheInfo;
using CacheItemId = uint64_t;
constexpr CacheItemId KInvalidCacheItemId = std::numeric_limits<uint64_t>::max();

using DelCacheFunc = std::function<bool(CacheInfo &)>;
using CCStatType = std::unordered_map<uint64_t, std::vector<CacheInfo>>;

class CacheInfo {
friend class CacheState;
public:
  CacheInfo(const uint64_t timer_count, const CacheItemId item_id, const CacheDescPtr &desc)
     : item_id_(item_id), desc_(desc), timer_count_(timer_count) {}
  CacheInfo(const CacheInfo &other)
     : item_id_(other.item_id_), desc_(other.desc_), timer_count_(other.timer_count_) {}
  CacheInfo &operator=(const CacheInfo &other) {
    timer_count_ = other.timer_count_;
    item_id_ = other.item_id_;
    desc_ = other.desc_;
    return *this;
  }
  CacheInfo() = delete;
  ~CacheInfo() = default;

  void RefreshTimerCount(uint64_t time_count) {
    timer_count_ = time_count;
  }

  uint64_t GetTimerCount() const noexcept {
    return timer_count_;
  }

  CacheItemId GetItemId() const noexcept {
    return item_id_;
  }

  const CacheDescPtr &GetCacheDesc() const noexcept {
    return desc_;
  }

private:
  CacheItemId item_id_;
  CacheDescPtr desc_;
  uint64_t timer_count_;
};

struct CacheInfoQueue {
  void Insert(const CacheHashKey main_hash_key, std::vector<CacheInfo> &cache_info);
  void EmplaceBack(const CacheHashKey main_hash_key, CacheInfo &cache_info);
  void Erase(std::vector<CacheItemId> &delete_ids, const DelCacheFunc &is_need_delete_func);

  CCStatType cc_state_;
  uint64_t cache_info_num_ = 0U;
};

class CacheState {
public:
  CacheState() = default;
  ~CacheState() = default;

  CacheItemId AddCache(const CacheHashKey main_hash_key, const CacheDescPtr &cache_desc);

  std::vector<CacheItemId> DelCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DelCache(const std::vector<CacheItemId> &delete_item);

  const CCStatType &GetState() const {
    return cache_info_queue.cc_state_;
  }

  uint64_t GetCacheInfoNum() const {
    return cache_info_queue.cache_info_num_;
  }

  uint64_t GetCurTimerCount() const {
    return cache_timer_count_;
  }
private:
  CacheItemId GetNextCacheItemId();
  void RecoveryCacheItemId(const std::vector<CacheItemId> &cache_items);
  uint64_t GetNextTimerCount() {
    const std::lock_guard<std::mutex> lock(cache_timer_count_mu_);
    return cache_timer_count_++;
  }

  std::mutex cache_info_queue_mu_;
  std::mutex cache_item_mu_;

  int64_t cache_item_counter_ = 0L;
  std::queue<int64_t> cache_item_queue_;
  CacheInfoQueue cache_info_queue;

  uint64_t cache_timer_count_ = 0U;
  std::mutex cache_timer_count_mu_;
};
}  // namespace ge
#endif