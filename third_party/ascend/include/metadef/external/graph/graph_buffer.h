/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

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
  GraphBuffer(const GraphBuffer &) = delete;
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
