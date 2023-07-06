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
#ifndef INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
#define INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/rt_ffts_plus.h"
#include "common/sgt_slice_type.h"
namespace ffts {
class FFTSPlusEngineUpdate {
public:
  FFTSPlusEngineUpdate();
  ~FFTSPlusEngineUpdate();
  static bool UpdateCommonCtx(ge::ComputeGraphPtr &sgt_graph, rtFftsPlusTaskInfo_t &task_info);
  static ThreadSliceMapDyPtr slice_info_ptr_;
};
};
#endif  // INC_REGISTER_FFTS_PLUS_ENGINE_UPDATE_H_
