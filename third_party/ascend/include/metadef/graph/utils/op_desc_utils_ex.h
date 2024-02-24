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

#ifndef __INC_METADEF_OP_DESC_UTILS_EX_H
#define __INC_METADEF_OP_DESC_UTILS_EX_H

#include "graph/op_desc.h"

namespace ge {
class OpDescUtilsEx {
 public:
  // Detach from OpDesc
  static graphStatus CallInferFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFormatFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferValueRangeFunc(const OpDescPtr &op_desc, Operator &op);
  static graphStatus OpVerify(const OpDescPtr &op_desc);
  static graphStatus InferShapeAndType(const OpDescPtr &op_desc);
  static graphStatus InferDataSlice(const OpDescPtr &op_desc);
  static void SetType(OpDescPtr &op_desc, const std::string &type);
  static void UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst);

 private:
  static graphStatus CallInferFuncV1(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFuncV2(const OpDescPtr &op_desc, Operator &op);
  static graphStatus CallInferFuncV2Inner(const OpDescPtr &op_desc, Operator &op);
  static graphStatus InferShapeByOutputShapesAttr(const OpDescPtr &op_desc);
};
} // namespace ge
#endif // __INC_METADEF_OP_DESC_UTILS_EX_H
