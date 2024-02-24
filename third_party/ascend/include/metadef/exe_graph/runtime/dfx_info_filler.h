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

#ifndef METADEF_INC_EXE_GRAPH_RUNTIME_DFX_INFO_FILLER_H
#define METADEF_INC_EXE_GRAPH_RUNTIME_DFX_INFO_FILLER_H

#include <string>
#include <vector>
#include <graph/ge_error_codes.h>

namespace gert {
class ProfilingInfoWrapper {
 public:
  virtual ~ProfilingInfoWrapper() = default;

  virtual void SetBlockDim(uint32_t block_dim) {
    (void)block_dim;
  }

  virtual void SetBlockDimForAtomic(uint32_t block_dim) {
    (void)block_dim;
  }

  virtual ge::graphStatus FillShapeInfo(const std::vector<std::vector<int64_t>> &input_shapes,
                                        const std::vector<std::vector<int64_t>> &output_shapes) {
    (void)input_shapes;
    (void)output_shapes;
    return ge::GRAPH_SUCCESS;
  }
};

class DataDumpInfoWrapper {
 public:
  virtual ~DataDumpInfoWrapper() = default;
  virtual ge::graphStatus CreateFftsCtxInfo(uint32_t thread_id, uint32_t context_id) = 0;
  virtual ge::graphStatus AddFftsCtxAddr(uint32_t thread_id, bool is_input, uint64_t address, uint64_t size) = 0;
  virtual void AddWorkspace(uintptr_t addr, int64_t bytes) = 0;
  virtual bool SetStrAttr(const std::string &name, const std::string &value) = 0;
};

class ExceptionDumpInfoWrapper {
 public:
  virtual ~ExceptionDumpInfoWrapper() = default;
  virtual void SetTilingData(uintptr_t addr, size_t size) = 0;
  virtual void SetTilingKey(uint32_t key) = 0;
  virtual void SetHostArgs(uintptr_t addr, size_t size) = 0;
  virtual void SetDeviceArgs(uintptr_t addr, size_t size) = 0;
  virtual void AddWorkspace(uintptr_t addr, int64_t bytes) = 0;
};
}

#endif

