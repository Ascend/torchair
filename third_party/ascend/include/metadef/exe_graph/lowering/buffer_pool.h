/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace gert {
namespace bg {
class BufferPool {
 public:
  using BufId = size_t;
  BufId AddStr(const char *data);
  BufId AddBuf(const uint8_t *data, const size_t len);
  std::unique_ptr<uint8_t[]> Serialize(size_t &total_size) const;
  std::unique_ptr<uint8_t[]> Serialize() const;
  size_t GetSize() const;

  // very slow, only use in UT
  const char *GetBufById(const BufId id) const;

 private:
  BufId AddBuf(std::string &&str);
  BufId AddLargeBuf(std::string &&str);

 private:
  std::unordered_map<std::string, BufId> bufs_to_id_;
  std::vector<std::pair<std::string, BufId>> large_bufs_to_id_; // large buf size, not do hash
  uint64_t id_generator_{0U};
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_BUFFER_POOL_H_
