/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include "ascir_register.h"
#include "graph/types.h"

namespace ascir {
EXPORT_GENERATOR()

// todo: to be deleted，无输入节点需要图提供特别的接口
REG_ASC_IR_1IO(Data).StartNode();
REG_ASC_IR_1IO(Workspace).StartNode();
REG_ASC_IR_1IO(Output);

REG_ASC_IR_1IO(Load).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_1IO(Broadcast).UseFirstInputDataType();
REG_ASC_IR_1IO(Store).UseFirstInputDataType().UseFirstInputView();
/*
 * todo nop比较特别，不确定是不是缺陷，原定义中，GEIR与ASCIR是不同的，GEIR多了个必选属性
namespace ge {
REG_OP(Nop)
    .REQUIRED_ATTR(dst_type, Int)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
.OP_END_FACTORY_REG(Nop)
}

namespace ascir::ops {
REG_OPS(Nop)
OPS_INPUT(0, x)
OPS_OUTPUT(0, y)
END_OPS(Nop)
}
 */
REG_ASC_IR_1IO(Nop).UseFirstInputDataType();
REG_ASC_IR_1IO(Cast)
    .Attr<ge::DataType>("dst_type")
    .UseFirstInputView()
    .InferDataType([](const AscIrDef &def, std::stringstream &ss) { ss << "  op.y.dtype = dst_type;" << std::endl; });

REG_ASC_IR_1IO(Abs).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_1IO(Max).UseFirstInputDataType();
REG_ASC_IR_1IO(Sum).UseFirstInputDataType();
REG_ASC_IR_2I1O(Sub).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_2I1O(Div).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_1IO(Exp).UseFirstInputDataType().UseFirstInputView();

REG_ASC_IR_2I1O(MatMul).UseFirstInputDataType();
REG_ASC_IR_2I1O(Muls).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_2I1O(Mul).UseFirstInputDataType().UseFirstInputView();
REG_ASC_IR_2I1O(Add).UseFirstInputDataType().UseFirstInputView();

REG_ASC_IR_START_NODE(TbufData);

REG_ASC_IR(FlashSoftmax).Inputs({"x1", "x2", "x3"}).Outputs({"y1", "y2", "y3"}).UseFirstInputDataType();
REG_ASC_IR_2I1O(Dropout).UseFirstInputDataType();
REG_ASC_IR_2I1O(Select).UseFirstInputDataType();
}  // namespace ascir