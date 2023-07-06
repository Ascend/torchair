/*
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

#ifndef GRAPH_COMPILE_CACHE_POLICY_HASH_UTILS_H_
#define GRAPH_COMPILE_CACHE_POLICY_HASH_UTILS_H_

#include <cstdint>
#include <vector>
#include <functional>
#include "graph/small_vector.h"
#include "graph/types.h"

namespace ge {
using CacheHashKey = uint64_t;
class HashUtils {
public:
  static constexpr CacheHashKey HASH_SEED = 0x7863a7deUL;
  static constexpr CacheHashKey COMBINE_KEY = 0x9e3779b9UL;

  template <typename T>
  static inline CacheHashKey HashCombine(CacheHashKey seed, const T &value) {
    const std::hash<T> hasher;
    seed ^= hasher(value) + COMBINE_KEY + (seed << 6U) + (seed >> 2U);
    return seed;
  }

  static inline CacheHashKey HashCombine(CacheHashKey seed, const Format &value) {
    const std::hash<int32_t> hasher;
    seed ^= hasher(static_cast<int32_t>(value)) + COMBINE_KEY + (seed << 6U) + (seed >> 2U);
    return seed;
  }

  static inline CacheHashKey HashCombine(CacheHashKey seed, const DataType &value) {
    const std::hash<int32_t> hasher;
    seed ^= hasher(static_cast<int32_t>(value)) + COMBINE_KEY + (seed << 6U) + (seed >> 2U);
    return seed;
  }

  template <typename T, size_t N>
  static inline CacheHashKey HashCombine(CacheHashKey seed, const SmallVector<T, N> &values) {
    for (const auto &val : values) {
      seed = HashCombine(seed, val);
    }
    return seed;
  }

  template <typename T>
  static inline CacheHashKey HashCombine(CacheHashKey seed, const std::vector<T> &values) {
    for (const auto &val : values) {
      seed = HashCombine(seed, val);
    }
    return seed;
  }

  static inline CacheHashKey MultiHash() {
    return HASH_SEED;
  }

  template <typename T, typename... M>
  static inline CacheHashKey MultiHash(const T &value, const M... args) {
    return HashCombine(MultiHash(args...), value);
  }
};
}  // namespace ge
#endif