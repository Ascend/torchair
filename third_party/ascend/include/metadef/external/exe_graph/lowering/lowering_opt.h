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

#ifndef INC_EXTERNAL_LOWERING_OPT_H_
#define INC_EXTERNAL_LOWERING_OPT_H_

#include <cstdint>

namespace gert {
struct LoweringOption {
  /**
   * 是否相信用户传入的输出tensor上的shape，如果开启了本选项，可以省去计算图上输出节点的InferShape，提升一点Host调度性能。
   * 与此同时，也会损失掉对外部传入的输出Tensor Shape、TensorData长度的校验能力。
   *
   * 约束：
   * 1. 如果一个节点有多个输出，并且部分输出并不是网络的输出，
   *    那么这个节点的InferShape不会被省掉，体现为在这个节点上，本选项会被忽略。
   * 2. 如果一个节点没有InferShape函数，例如第三类、第四类算子，
   *    需要从Device拷贝回Shape，那么在这个节点上，本选项会被忽略。
   * 3. 本选项是个加载时选项，一旦选定后，意味着后续本model的每次调用都需要用户传入输出shape，否则可能会导致执行失败
   */
  bool trust_shape_on_out_tensor = false;

  /**
   * 总是零拷贝开关，默认关闭。如果本开关打开，含义是外部调用者总是保证会正确地申请输出内存，包含：
   * 1. 申请的输出内存大于等于输出shape所以计算出的Tensor大小
   * 2. 输出内存的placement正确
   *
   * 打开本开关后，可以提升一点Host调度性能。与此同时，对于零拷贝失效的回退处理将不再进行，
   * 在外部申请的输出内存错误、或未申请输出内存时，执行报错。
   */
  bool always_zero_copy = false;

  /**
   * 总是使用外部allocator开关，默认关闭。如果本开关打开，含义是外部调用者总是保证会传入所有allocator，包含：
   * 1. 创建所有在加载/执行阶段所需的allocator并传入
   * 2. 由于总是信任外部allocator，一旦开启后，如果在加载/执行阶段获取外置allocator失败，则报错。
   *
   * 打开本开关后，在执行器内部不需要再创建allocator，减少资源浪费
   */
  bool always_external_allocator = false;

  /**
   * 二进制兼容保留字段，增加option时，对应缩减删除reserved长度
   */
  uint8_t reserved[5U + 8U] = {0U};
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GRAPH_H_
