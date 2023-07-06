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

#ifndef COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
#define COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_

#include <memory>
#include "graph/types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "transfer_shape_utils.h"

namespace transformer {
struct CalcShapeExtraAttr {
  int64_t hidden_size;
  int64_t input_size;
  int64_t state_size;
};

struct ShapeAndFormatInfo {
  ge::GeShape &oldShape;
  const ge::Format &oldFormat;
  const ge::Format &newFormat;
  const ge::DataType &currentDataType;
  CalcShapeExtraAttr extra_attr;
  ShapeAndFormatInfo(ge::GeShape &old_shape, const ge::Format &old_format, const ge::Format &new_format,
                     const ge::DataType &data_type)
                     : oldShape(old_shape), oldFormat(old_format), newFormat(new_format), currentDataType(data_type),
                       extra_attr({1, 1, -1}) {}
};

using ShapeAndFormat = struct ShapeAndFormatInfo;

class ShapeTransferAccordingToFormat {
 public:
  ShapeTransferAccordingToFormat();

  ~ShapeTransferAccordingToFormat() {};

  ShapeTransferAccordingToFormat(const ShapeTransferAccordingToFormat&) = delete;

  ShapeTransferAccordingToFormat &operator=(const ShapeTransferAccordingToFormat&) = delete;

  static bool GetShapeAccordingToFormat(ShapeAndFormat &shapeAndFormatInfo);

  static bool GetShapeAccordingToFormat(const ge::OpDescPtr &op_desc, ShapeAndFormat &shapeAndFormatInfo);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, ge::GeShape &shape);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, const ge::GeShape &origin_shape, ge::GeShape &shape);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            gert::Shape &shape, const ge::OpDescPtr op_desc = nullptr);

  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const gert::Shape &origin_shape, gert::Shape &shape, const ge::OpDescPtr op_desc = nullptr);

  static void InitExtAxisValue(const ge::OpDescPtr &op_desc, ExtAxisValue &ext_axis);
};
} // namespace transformer
#endif  // COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
