/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef INC_EXTERNAL_GRAPH_BUFFER_H_
#define INC_EXTERNAL_GRAPH_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include "./types.h"

namespace ge {
class Graph;
class Buffer;
using BufferPtr = std::shared_ptr<Buffer>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GraphBuffer {
 public:
  GraphBuffer();
  GraphBuffer(const GraphBuffer &) = delete;;
  GraphBuffer &operator=(const GraphBuffer &) = delete;
  ~GraphBuffer();

  const std::uint8_t *GetData() const;
  std::size_t GetSize() const;

 private:
  BufferPtr buffer_{nullptr};
  friend class Graph;
};
}  // namespace ge
#endif  // INC_EXTERNAL_GRAPH_BUFFER_H_
