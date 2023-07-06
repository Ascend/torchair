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

#ifndef INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_
#define INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_

#include <cstdint>

namespace ge {
namespace dflow {
struct CountBatch {
  int64_t batch_size = 0L;
  int64_t slide_stride = 0L;
  int64_t timeout = 0L;
  int64_t batch_dim = 0L;
  int32_t flag = 0; // eg: eos/seg
  bool padding = false;
  bool drop_remainder = false;
  int8_t rsv[90] = {};
};

struct TimeBatch {
  int64_t time_window = 0L;
  int64_t time_interval = 0L;
  int64_t timeout = 0L;
  int64_t batch_dim = -1;
  int32_t flag = 0; // eg: eos/seg
  bool padding = false;
  bool drop_remainder = false;
  int8_t rsv[90] = {};
};

enum class DataFlowAttrType {
  COUNT_BATCH = 0,
  TIME_BATCH = 1,
  INVALID = 2,
};

struct DataFlowInputAttr {
  DataFlowAttrType attr_type;
  void *attr_value;
};
} // namespace dflow
} // namespace ge
#endif // INC_EXTERNAL_FLOW_GRAPH_FLOW_ATTR_H_