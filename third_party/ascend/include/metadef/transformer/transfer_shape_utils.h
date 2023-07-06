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

#ifndef TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_
#define TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_

#include <array>
#include "axis_util.h"

namespace transformer {
const size_t ORIGIN_FORMAT_DIM_SIZE = 5;
const size_t EXT_AXIS_SIZE = 4;
using FormatIndex = std::array<size_t, ORIGIN_FORMAT_DIM_SIZE>;
using ExtAxisValue = std::array<int64_t, EXT_AXIS_SIZE>;

class TransferShapeUtils {
public:
  TransferShapeUtils() {}
  ~TransferShapeUtils() {}
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, gert::Shape &shape);
  static bool TransferShape(const ge::Format &origin_format, const ge::Format &format, const ge::DataType &data_type,
                            const ExtAxisValue &ext_axis, const gert::Shape &origin_shape, gert::Shape &shape);

private:
  static bool TransferShapeByFormat(const ge::Format &primary_format, const AxisValue &axis_value,
                                    gert::Shape &shape);
  static bool TransferShapeByAxisValue(const ge::Format &primary_format, const AxisValue &axis_value,
                                       gert::Shape &shape);
  static bool TransferShapeByOriginShape(const ge::Format &primary_format, const int64_t &c0,
                                         const ExtAxisValue &ext_axis, const gert::Shape &origin_shape,
                                         gert::Shape &shape);
  static bool TransferShapeByFormatIndex(const ge::Format &origin_format, const ge::Format &format, const int64_t &c0,
                                         const gert::Shape &origin_shape, gert::Shape &shape);
  static bool IsNeedTransferShape(const ge::Format &origin_format, const ge::Format &format, const gert::Shape &shape);
  static bool CheckInputParam(const ge::Format &origin_format, const ge::Format &primary_format,
                              const ge::DataType &data_type);
  static bool IsNeedAxisValue(const ge::Format &format, const size_t &origin_dim_size);
  static int64_t GetC0Value(const ge::DataType &data_type, const ge::Format &format);

  /* ----------Below is the function of getting new shape by axis value---------------------- */
  static bool GetNCHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetCHWNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNDC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetC1HWNCoC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFz3DShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFz3DTransposeShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzLstmShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFzC04ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetFznRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  static bool GetNDRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape);

  /* ----------Below is the function of getting new shape by origin shape---------------------- */
  static bool GetNCHWShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNHWCShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetHWCNShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetCHWNShape(const FormatIndex& format_index, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetNC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                              gert::Shape &shape);

  static bool GetNDC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                               gert::Shape &shape);

  static bool GetC1HWNCoC0Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                                gert::Shape &shape);

  static bool GetFractalNzShape(const ExtAxisValue &ext_axis, const int64_t &c0, const gert::Shape &origin_shape,
                                gert::Shape &shape);

  static bool GetFractalZShape(const int64_t &c0, const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                               const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZ3DShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                                 const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZ3DTransposeShape(const FormatIndex& format_index, const int64_t &c0,
                                          const gert::Shape &origin_shape, gert::Shape &shape);

  static bool GetFractalZLstmShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                   gert::Shape &shape);

  static bool GetFractalZC04Shape(const FormatIndex& format_index, const int64_t &c0, const gert::Shape &origin_shape,
                                  gert::Shape &shape);

  static bool GetFractalZnRnnShape(const ExtAxisValue &ext_axis, const int64_t &c0, const gert::Shape &origin_shape,
                                   gert::Shape &shape);

  static bool GetNdRnnBiasShape(const ExtAxisValue &ext_axis, const int64_t &c0, const gert::Shape &origin_shape,
                                gert::Shape &shape);

  static bool GetNYUVShape(gert::Shape &shape);
};
}
#endif  // TRANSFORMER_INC_TRANSFER_SHAPE_UTILS_H_
