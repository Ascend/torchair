/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef INC_COMMON_LARGE_BM_H_
#define INC_COMMON_LARGE_BM_H_

#include <vector>
#include <memory>

/* LargeBitmap create a way to generate bitmaps larger than 64bit. */
namespace ge {
class LargeBitmap {
public:
  explicit LargeBitmap(const size_t &size);

  ~LargeBitmap() = default;

  bool operator==(const LargeBitmap &another_bm) const;

  bool operator!=(const LargeBitmap &another_bm) const;

  // set all vector to specific value
  void SetValues(const uint64_t &value);

  // Get the value on position index
  bool GetBit(const size_t &index) const;

  // Set the value on position index to 1
  void SetBit(const size_t &index);

  // Combine two bitmap with the following rule.
  // If one bit of either one of the two bitmaps is 1,
  // the result of final bitmap is 1.
  void Or(const LargeBitmap &another_bm);

  // Combine two bitmap with the following rule.
  // If one bit of either one of the two bitmaps is 0,
  // the result of final bitmap is 0.
  void And(const LargeBitmap &another_bm);

  void ClearBit(size_t bit_idx);

  void ResizeBits(size_t new_size);
private:
  // Number of element in vector bits
  size_t size_;

  std::vector<uint64_t> bits_;
};
}
#endif // INC_COMMON_LARGE_BM_H_
