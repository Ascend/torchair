#include "codegen_kernel.h"

#include <sstream>

#include "ascir_ops.h"
#include "codegen_common.h"

using namespace std;
using namespace ascir::ops;
using namespace codegen;

std::ostream& operator<<(std::ostream &os, const Code &code) {
  return os << code.Str();
}

Type::Type(const string& name) : name(name) {}

std::string Type::Str() const {
  return name;
}

Variable::Variable(const Type &type, const string &name) : type(type), name(name) {}

std::string Variable::Str() const {
  return name;
}

std::string Variable::AsArg() const {
  stringstream ss;
  ss << this->type << " " << this->name;
  return ss.str();
}

std::string Variable::Define(std::string &&init, bool define_const) const {
    std::stringstream ss;
    if (define_const) {
      ss << "const ";
    }

    if (init.empty()) {
      ss << type << " " << name << ";";
    } else {
      ss << type << " " << name << " = " << std::move(init) << ";";
    }
    return ss.str();
}

Axis::Axis(const ascir::Axis& axis)
    : ascir::Axis(axis), Variable(Int_t, axis.name), loop_size(Variable(Int_t, axis.name + "_loop_size")) {}

const std::string &Tensor::DtypeName(ge::DataType dtype) {
  static const std::string type_names[] = {
    [ge::DT_FLOAT] = "float",
    [ge::DT_FLOAT16] = "half"
  };

  if (dtype > sizeof(type_names) / sizeof(type_names[0])) {
      throw std::runtime_error("Unsupported data type");
  }

  return type_names[dtype];
}

const Type &Tensor::GlobalTensorTypes(ge::DataType dtype) {
  static const Type types[] = {
      [ge::DT_FLOAT] = Type("GlobalTensor<float>"),
      [ge::DT_FLOAT16] = Type("GlobalTensor<half>"),
  };

  if (dtype > sizeof(types) / sizeof(types[0])) {
      throw std::runtime_error("Unsupported data type");
  }

  return types[dtype];
}

const Type &Tensor::LocalTensorTypes(ge::DataType dtype) {
  static const Type types[] = {
      [ge::DT_FLOAT] = Type("LocalTensor<float>"),
      [ge::DT_FLOAT16] = Type("LocalTensor<half>"),
  };

  if (dtype > sizeof(types) / sizeof(types[0])) {
      throw std::runtime_error("Unsupported data type");
  }

  return types[dtype];
}

Tensor::Tensor(const ascir::TensorAttr &tensor, const std::string &name)
    : Variable(tensor.mem.alloc_type == ascir::ALLOC_TYPE_GLOBAL ? GlobalTensorTypes(tensor.dtype)
                                                                 : LocalTensorTypes(tensor.dtype),
               name.empty() ? "t" + to_string(tensor.mem.tensor_id) : name),
      id(tensor.mem.tensor_id),
      dtype(tensor.dtype),
      alloc_type(tensor.mem.alloc_type),
      position(tensor.mem.position),
      axis(tensor.axis),
      vectorized_axis(tensor.vectorized_axis),
      axis_size(tensor.repeats()),
      axis_strides(tensor.strides()),
      que_id(tensor.que.id),
      buf_id(tensor.buf.id),
      size(this->name + "_size"),
      que_depth(this->name + "_que_depth"),
      que_buf_num(this->name + "_que_buf_num"),
      que_depth_value(tensor.que.depth),
      que_buf_num_value(tensor.que.buf_num),
      merge_scope(tensor.opt.merge_scope) {
  for (auto vec_axis : this->vectorized_axis) {
    auto pos = std::find(this->axis.begin(), this->axis.end(), vec_axis);
    if (pos == this->axis.end()) {
      throw std::runtime_error("Vectorized axis " + to_string(vec_axis) + " not found.");
    }

    this->vectorized_axis_pos.push_back(pos - this->axis.begin());
  }
}

std::string Tensor::SetGlobalBuffer(GM_ADDR global) const {
  std::stringstream ss;
  ss << name << ".SetGlobalBuffer("
     << "(__gm__ " << DtypeName(this->dtype) << "*)" << global << ");";
  return ss.str();
}

std::string codegen::PositionValue(ascir::Position position) {
  const std::string positionValues[] = {
    [ascir::POSITION_GM] = "TPosition::GM",
    [ascir::POSITION_VECIN] = "TPosition::VECIN",
    [ascir::POSITION_VECOUT] = "TPosition::VECOUT",
  };

  if (position >= sizeof(positionValues) / sizeof(positionValues[0])) {
    throw std::invalid_argument("Position value invalid " + to_string(position));
  }

  return positionValues[position];
}

MergeScope::MergeScope(ascir::MergeScopeId id, ascir::Position position)
    : id(id),
      position(position),
      size("m" + to_string(id) + "_size"),
      depth("m" + to_string(id) + "_que_depth"),
      buf_num("m" + to_string(id) + "_que_buf_num") {}

TQue::TQue(ascir::QueId id, ascir::Position position)
    : Variable(Type("TQue<" + PositionValue(position) + ", " + "q" + to_string(id) + "_depth>"), "q" + to_string(id)),
      id(id),
      position(position),
      size(this->name + "_size"),
      depth(this->name + "_depth"),
      buf_num(this->name + "_buf_num"),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf") {}

std::string TQue::AllocBuf() const {
  stringstream ss;
  ss << this->buf.AsArg() << " = " << this->name << ".AllocTensor<uint8_t>();";
  return ss.str();
}

std::string TQue::FreeBuf() const {
  stringstream ss;
  ss << this->name << ".FreeTensor(" << this->buf << ");";
  return ss.str();
}

std::string TQue::EnqueBuf() const {
  stringstream ss;
  ss << this->name << ".EnQue(" << this->buf << ");";
  return ss.str();
}

std::string TQue::DequeBuf() const {
  stringstream ss;
  ss << this->buf.AsArg() << " = "  << this->name << ".DeQue<uint8_t>();";
  return ss.str();
}

TBuf::TBuf(ascir::BufId id, const ascir::Position position)
    : Variable(Type("TBuf<" + PositionValue(position) + ">"), "b" + to_string(id)),
      buf(Type("LocalTensor<uint8_t>"), name + "_buf"),
      id(id),
      position(position),
      size(this->name + "_size") {}

std::string TBuf::AllocBuf() const {
  stringstream ss;
  ss << this->buf.AsArg() << " = " << this->name << ".Get<uint8_t>();";
  return ss.str();
}

Tiler::Tiler(const std::string &tiling_data_name)
    : tiling_data(Type{"optiling::TilingData"}, tiling_data_name),
      block_dim("block_dim")
{}

std::string Tiler::Offset(const std::vector<ascir::AxisId> &current_axis, const std::vector<ascir::AxisId> &axis,
                          const std::vector<ascir::SizeExpr> &strides) const {
  std::stringstream ss;
  bool is_first = true;

  for (auto a : current_axis) {
    auto iter = std::find(axis.begin(), axis.end(), a);
    if (iter == axis.end()) {
      continue;
    }

    if (is_first) {
      is_first = false;
    } else {
      ss << " + ";
    }

    auto stride = strides[iter - axis.begin()];
    if (stride == 0) {
      ss << "0";
    } else if (stride == 1) {
      ss << this->GetAxis(a);
    } else {
      ss << this->GetAxis(a) << " * " << this->Size(stride);
    }
  }

  if (is_first) {
    // Not axis in current_axis
    ss << "0";
  }
  return ss.str();
}

std::string Tiler::BlkLen(ge::DataType dtype, const ascir::SizeExpr &size) const {
  std::stringstream ss;
  ss << KernelUtils::BlkLen(dtype) << "(" << this->Size(size) << ")";
  return ss.str();
}

std::string Tiler::BlkLenDiff(ge::DataType dtype, const ascir::SizeExpr &size_a, const ascir::SizeExpr &size_b) const {
  std::stringstream ss;
  ss << KernelUtils::BlkLen(dtype) << "(" << this->Size(size_a) << " - " << this->Size(size_b) << ")";
  return ss.str();
}

std::string Tiler::TensorVectorizedOffset(const std::vector<ascir::AxisId> &current_axis, const Tensor &tensor) const {
  std::vector<ascir::AxisId> current_vectorized_axis;
  for (auto a : current_axis) {
    if (find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), a)!= tensor.vectorized_axis.end()) {
      current_vectorized_axis.emplace_back(a);
    }
  }
  return this->Offset(current_vectorized_axis, tensor.axis, tensor.axis_strides);
}

std::string Tiler::Str() const {
  return tiling_data.Str();
}

void codegen::Tiler::AddSizeVar(const ascir::SizeVar &size) {
  auto [new_size_var, insert_success] = this->sizes.emplace(size.id, size);
  if (!insert_success) {
    throw std::invalid_argument("Duplicate size var id " + to_string(size.id));
  }
}

void codegen::Tiler::AddAxis(const ascir::Axis &axis) {
  auto [new_axis, insert_success] = this->axis.emplace(axis.id, codegen::Axis(axis));
  if (!insert_success) {
    throw std::invalid_argument("Duplicate axis id " + to_string(axis.id));
  }
}

std::string codegen::Tiler::Size(const ascir::SizeExpr &size) const {
  if (size.is_zero) {
    return "0";
  }

  stringstream ss;
  auto Muls = [&](typeof(size.nums) &size_list) -> std::stringstream & {
    if (size_list.size() == 0) {
      ss << "1";
    } else {
      if (size_list.size() > 1) {
        ss << "(";
      }

      for (size_t i = 0; i < size_list.size() - 1; i++) {
        auto size = this->sizes.find(size_list[i]);
        if (size == this->sizes.end()) {
            throw std::runtime_error("Size not found: " + std::to_string(size_list[i]));
        }
        ss << tiling_data << "." << size->second.name << " * ";
      }

      auto size = this->sizes.find(size_list.back());
      if (size == this->sizes.end()) {
        throw std::runtime_error("Size not found: " + std::to_string(size_list.back()));
      }
      ss << tiling_data << "." << size->second.name;

      if (size_list.size() > 1) {
        ss << ")";
      }
    }
    return ss;
  };

  if (size.dens.size() == 0) {
    Muls(size.nums);
  } else {
    Muls(size.nums);
    ss << " / ";
    Muls(size.dens);
  }
  return ss.str();
}

std::string Tiler::TensorVectorizedSize(const Tensor &tensor) const {
  stringstream ss;
  bool first = true;

  for (auto axis_pos : tensor.vectorized_axis_pos) {
    if (first) {
      first = false;
    } else {
      ss << " + ";
    }

    ss << "(" + this->Size(tensor.axis_size[axis_pos]) + " - 1)";
    if (!(tensor.axis_strides[axis_pos] == 1)) {
        ss << " * " << this->Size(tensor.axis_strides[axis_pos]);
    }
  }

  if (first) {
    ss << "1";
  } else {
    ss << " + 1";
  }
  return ss.str();
}

const Axis &Tiler::GetAxis(const ascir::AxisId id) const {
  auto iter = this->axis.find(id);
  if (iter == this->axis.end()) {
    throw std::runtime_error("Axis not found " + to_string(id));
  }

  return iter->second;
}

std::string codegen::Tiler::AxisSize(const ascir::AxisId id) const {
  return this->Size(this->GetAxis(id).size);
}

std::string codegen::Tiler::AxisSize(const Axis &axis) const {
  return this->Size(axis.size);
}

std::string codegen::Tiler::TilingDataDefine(GM_ADDR tiling_data_arg) const {
  std::stringstream ss;
  ss <<  "GET_TILING_DATA(" << this->tiling_data << ", " << tiling_data_arg << ");" << std::endl;
  return ss.str();
}

std::string codegen::Tiler::LoopSizeCal() const {
  std::stringstream ss;
  ascir::AxisId outer_axis_id = ascir::ID_NONE;
  ascir::AxisId inner_axis_id = ascir::ID_NONE;

  for (auto& [out_id, out_axis] : this->axis) {
    if (out_axis.type != ascir::Axis::AXIS_TYPE_BLOCK_OUTER) { continue; }
    outer_axis_id = out_id;

    inner_axis_id = ascir::ID_NONE;
    for (auto& [in_id, in_axis] : this->axis) {
      if (in_axis.type == ascir::Axis::AXIS_TYPE_BLOCK_INNER && in_axis.from == out_axis.from) {
        inner_axis_id = in_id;
        break;
      }
    }

    if (inner_axis_id == ascir::ID_NONE) {
      throw std::runtime_error("out axis " + out_axis.Str() + "has no relative block inner axis");
    }
    std::string from = this->AxisSize(GetAxis(GetAxis(outer_axis_id).from[0]));
    std::string outter_axis = GetAxis(outer_axis_id).Str();
    std::string origin_axis_size = this->AxisSize(GetAxis(GetAxis(outer_axis_id).from[0]));
    std::string inner_axis_size = this->AxisSize(GetAxis(inner_axis_id));
    std::string loop_size = "(" + outter_axis + " + 1 ) * " + inner_axis_size + " > " + origin_axis_size
                            + " ? " + origin_axis_size + " - " + outter_axis + " * " + inner_axis_size
                            + " : " + inner_axis_size;
    ss << GetAxis(inner_axis_id).loop_size.DefineConst(move(loop_size)) << std::endl;
    // 这里的break是因为假设只有一个Block_outter轴，支持多个后就可以去掉
    break;
  }
  return ss.str();
}

std::string codegen::Tiler::BlockOutterAxisDefine() {
  stringstream code;

  code << this->block_dim.Define("GetBlockIdx()") << std::endl;

  for (auto& [id, axis] : this->axis) {
    if (axis.type != ascir::Axis::AXIS_TYPE_BLOCK_OUTER) {
      continue;
    }

    stringstream axis_value;
    axis_value << this->block_dim.name << " % (utils::Ceil(1.0 * " << this->AxisSize(axis) << "))";
    code << axis.Define(axis_value.str(), true);
    code << " ";

    code << this->block_dim.name << " /= " << this->AxisSize(axis) << ";";
    code << std::endl;
  }

  return code.str();
}

std::string AscendCApiExtend::GetSetTmp() {
  return std::string {
    "LocalTensor<uint8_t> g_local_tmp;\n"
    "void SetTmp(LocalTensor<uint8_t> tmp){\n"
    "    g_local_tmp = tmp;\n"
    "};\n"
    "\n"
    "template <typename T>\n"
    "LocalTensor<T> GetTmp() {\n"
    "    LocalTensor<T> tmp;\n"
    "    tmp.SetAddrWithOffset(g_local_tmp, 0);\n"
    "    return tmp;\n"
    "}\n"
  };
}

std::string codegen::AscendCApiExtend::ReduceLast() {
  return std::string{
    "template<typename T>\n"
    "void WholeReduceSumAdapt(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const int32_t mask,\n"
    "                         const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,\n"
    "                         const int32_t srcRepStride, ReduceOrder order){\n"
    "    WholeReduceSum(dstLocal, srcLocal, mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);\n"
    "}\n"
    "\n"
    "template <typename T,\n"
    "          void (*WholeReduceFunc)(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const int32_t mask,\n"
    "                                  const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,\n"
    "                                  const int32_t srcRepStride, ReduceOrder order),\n"
    "          void (*BinaryFunc)(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,\n"
    "                             const LocalTensor<T> &src1Local, const int32_t &calCount)>\n"
    "void ReduceLast(const LocalTensor<T> &dst, const LocalTensor<T> &src, const int32_t m, const int32_t k) {\n"
    "    if (k <= FULL_MASK_LEN && (k * sizeof(half) % ONE_BLK_SIZE == 0)) {\n"
    "      int32_t srcRepStride = (k * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE;\n"
    "      WholeReduceFunc(dst, src, k, m, 1, 1, srcRepStride, ReduceOrder::ORDER_ONLY_VALUE);\n"
    "    } else if (k > FULL_MASK_LEN && k % FULL_MASK_LEN == 0 && m < FULL_MASK_LEN) {\n"
    "      WholeReduceFunc(dst, src, FULL_MASK_LEN, m, 1, 1, utils::BlkLen<T>(k), ReduceOrder::ORDER_ONLY_VALUE);\n"
    "\n"
    "      auto tmp_buf = GetTmp<T>();\n"
    "      for (int i = FULL_MASK_LEN; i < k; i += FULL_MASK_LEN) {\n"
    "        pipe_barrier(PIPE_V);\n"
    "        WholeReduceFunc(tmp_buf, src[i], FULL_MASK_LEN, m, 1, 1, utils::BlkLen<T>(k), ReduceOrder::ORDER_ONLY_VALUE);\n"
    "        pipe_barrier(PIPE_V);\n"
    "        BinaryFunc(dst, dst, tmp_buf, m);\n"
    "      }\n"
    "    } else {\n"
    "      ASSERT(false && \"Reduce k size not support.\");\n"
    "    }\n"
    "}\n"
  };
}

std::string AscendCApiExtend::FunctionDefines() {
  std::stringstream ss;
  ss << "/* AscendC Api Extend */" << std::endl;
  ss << GetSetTmp() << std::endl;
  ss << ReduceLast() << std::endl;
  return ss.str();
}

std::string KernelUtils::FunctionDefines() {
  std::stringstream ss;

  ss << "namespace utils {" << std::endl;
  ss << "int Ceil(float num) {" << std::endl;
  ss << "int ceil_num = (int)num;" << std::endl;
  ss << "return num == (float)ceil_num ? ceil_num : (ceil_num + 1);" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "template <typename T>" << std::endl;
  ss << "constexpr inline __aicore__ T Max(const T a) {" << std::endl;
  ss << "  return a;" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "template <typename T, typename... Ts>" << std::endl;
  ss << "constexpr inline __aicore__ T Max(const T a, const Ts... ts) {" << std::endl;
  ss << "  return a > Max(ts...) ? a : Max(ts...);" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "template <typename T, typename... Ts>" << std::endl;
  ss << "constexpr inline __aicore__ T Sum(const T a, const Ts... ts) {" << std::endl;
  ss << "  return (a + ... + ts);" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "template<typename DataType>" << std::endl;
  ss << "constexpr inline __ai_core__ uint16_t BlkLen(uint32_t size) {" << std::endl;
  ss << "  return size / (ONE_BLK_SIZE / sizeof(DataType));" << std::endl;
  ss << "}" << std::endl;
  ss << std::endl;
  ss << "constexpr inline __ai_core__ uint16_t BlkAlign(uint32_t size) {" << std::endl;
  ss << "  return (size + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;

  return ss.str();
}

std::string KernelUtils::Max() {
  return "utils::Max";
}

std::string KernelUtils::Sum() {
  return "utils::Sum";
}

std::string KernelUtils::BlkLen(ge::DataType dtype) {
  std::stringstream ss;
  ss << "utils::BlkLen<" << Tensor::DtypeName(dtype) << ">";
  return ss.str();
}

std::string KernelUtils::BlkAlign() {
  return "utils::BlkAlign";
}

TPipe::TPipe(const std::string &name, const Tiler &tiler)
    : Variable(Type{"TPipe"}, name), tiler(tiler) {}

Tensor& TPipe::AddTensor(const Tensor& tensor) {
  auto [ret, is_insert] = this->tensors.emplace(tensor.id, tensor);
  if (!is_insert) {
    throw std::invalid_argument("Tensor is already added " + to_string(tensor.id) + " " + tensor.name);
  }

  auto &t = ret->second;
  if (t.merge_scope != ascir::ID_NONE && t.alloc_type != ascir::ALLOC_TYPE_GLOBAL) {
    auto merge_scope = this->merge_scopes.find(t.merge_scope);
    if (merge_scope == this->merge_scopes.end()) {
      auto [new_scope, is_insert] = this->merge_scopes.emplace(t.merge_scope, MergeScope{t.merge_scope, t.position});
      new_scope->second.tensors.push_back(t.id);
    } else {
      if (merge_scope->second.position != t.position) {
        throw std::runtime_error("Merge scope position mismatch between " + to_string(t.position) + " and " +
                                 to_string(merge_scope->second.position));
      }
      merge_scope->second.tensors.push_back(t.id);
    }
  }

  if (t.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
    if (t.que_id == ascir::ID_NONE) {
        throw std::runtime_error("Tensor queue is none " + to_string(t.id) + " " + t.name);
    }

    TQue* que = nullptr;
    auto iter = this->ques.find(t.que_id);
    if (iter == this->ques.end()) {
        auto [new_que, is_insert] = this->ques.emplace(t.que_id, TQue{t.que_id, t.position});
        que = &new_que->second;
    } else {
        que = &iter->second;
    }

    if (que->position != t.position) {
        throw std::runtime_error("Que position missmatch for " + t.name + " between " + to_string(t.position) +
                                 " and " + to_string(que->position));
    }

    if (t.merge_scope != ascir::ID_NONE) {
        que->merge_scopes.insert(t.merge_scope);
    } else {
        que->not_merge_tensors.insert(t.id);
    }
  } else if (t.alloc_type == ascir::ALLOC_TYPE_BUFFER) {
    if (t.buf_id == ascir::ID_NONE) {
        throw std::runtime_error("Tensor buffer is None " + to_string(t.id) + " " + t.name);
    }

    TBuf *buf = nullptr;
    auto iter = this->bufs.find(t.buf_id);
    if (iter == this->bufs.end()) {
        auto [new_buf, is_insert] = this->bufs.emplace(t.buf_id, TBuf{t.buf_id, t.position});
        buf = &new_buf->second;
    } else {
        buf = &iter->second;
    }

    if (buf->position != t.position) {
        throw std::runtime_error("Buf position mismatch for " + t.name + " between " + to_string(t.position) + " and " +
                                 to_string(buf->position));
    }

    if (t.merge_scope != ascir::ID_NONE) {
        buf->merge_scopes.insert(t.merge_scope);
    } else {
        buf->not_merge_tensors.insert(t.id);
    }
  }

  return t;
}

Tensor& TPipe::AddTensor(const ascir::TensorAttr &tensor, const std::string &name) {
  return this->AddTensor(Tensor(tensor, name));
}

const TQue &TPipe::GetQue(const ascir::QueId id) const {
  auto iter = this->ques.find(id);
  if (iter == this->ques.end()) {
    throw std::runtime_error("Que not found " + to_string(id));
  }

  return iter->second;
}

const TBuf &TPipe::GetBuf(const ascir::BufId id) const {
  auto iter = this->bufs.find(id);
  if (iter == this->bufs.end()) {
    throw std::runtime_error("Buf not found " + to_string(id));
  }

  return iter->second;
}

const Tensor &TPipe::GetTensor(const ascir::TensorId id) const {
  auto iter = this->tensors.find(id);
  if (iter == this->tensors.end()) {
    throw std::runtime_error("Tensor not found " + to_string(id));
  }

  return iter->second;
}

std::string TPipe::TensorAlloc(const Tensor& tensor) const {
  std::stringstream ss;
  ss << tensor.Define() << std::endl;

  const Variable* buf;
  if (tensor.alloc_type == ascir::ALLOC_TYPE_BUFFER) {
      buf = &GetBuf(tensor.buf_id).buf;
  } else if (tensor.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
      buf = &GetQue(tensor.que_id).buf;
  } else if (tensor.alloc_type == ascir::ALLOC_TYPE_GLOBAL) {
      buf = &tensor;
  } else {
      throw std::runtime_error("Tensor alloc type not supported " + to_string(tensor.alloc_type));
  }

  if (tensor.merge_scope == ascir::ID_NONE) {
      ss << tensor << ".SetAddrWithOffset(" << *buf << ", 0);" << std::endl;
  } else {
      throw std::runtime_error("Tensor merge scope not supported " + to_string(tensor.merge_scope));
  }

  return ss.str();
}

std::string TPipe::InitTQueBuffers(const TQue &que) const {
  stringstream ss;
  ss << this->name << "."
     << "InitBuffer(" << que << ", " << que.buf_num << ", " << KernelUtils::BlkAlign() << "(" << que.size << "));";
  return ss.str();
}

std::string TPipe::InitTBufBuffer(const TBuf &buf) const {
  stringstream ss;
  ss << this->name << "."
     << "InitBuffer(" << buf << ", " << KernelUtils::BlkAlign() << "(" << buf.size << "));";
  return ss.str();
}

std::string TPipe::TensorSizeCalc() const {
  stringstream ss;

  for (auto& [id, t] : this->tensors) {
    if (t.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
      ss << t.que_depth.DefineConst(to_string(t.que_depth_value)) << std::endl;
      ss << t.que_buf_num.DefineConst(to_string(t.que_buf_num_value)) << std::endl;
    } else if (t.alloc_type == ascir::ALLOC_TYPE_BUFFER) {
      ss << t.size.DefineConst(this->tiler.TensorVectorizedSize(t)) << std::endl;
    }
  }

  return ss.str();
}

std::string TPipe::MergeScopeSizeCalc() const {
  stringstream ss;

  for (auto &[id, merge_scope] : this->merge_scopes) {
    stringstream tensor_size_sum;
    stringstream tensor_depth_max;
    stringstream tensor_bufnum_max;

    tensor_size_sum << KernelUtils::Sum() << "(";
    tensor_depth_max << KernelUtils::Max() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool first = true;
    for (auto tid : merge_scope.tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        throw std::runtime_error("Tensor not found: " + std::to_string(tid));
      }

      if (tensor->second.alloc_type != ascir::ALLOC_TYPE_QUEUE &&
          tensor->second.alloc_type != ascir::ALLOC_TYPE_BUFFER) {
        throw std::runtime_error("Tensor " + tensor->second.name + " is not alloc from que/buf");
      }

      if (first) {
        first = false;
      } else {
        tensor_size_sum << ", ";
        if (tensor->second.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
          tensor_depth_max << ", ";
          tensor_bufnum_max << ", ";
        }
      }

      tensor_size_sum << tensor->second.size << " * " << "sizeof(" << Tensor::DtypeName(tensor->second.dtype) << ")";
      if (tensor->second.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
        tensor_depth_max << tensor->second.que_depth;
        tensor_bufnum_max << tensor->second.que_buf_num;
      }
    }

    tensor_size_sum << ")";
    tensor_depth_max << ")";
    tensor_bufnum_max << ")";

    ss << merge_scope.size.DefineConst(tensor_size_sum.str()) << std::endl;
    ss << merge_scope.depth.DefineConst(tensor_depth_max.str()) << std::endl;
    ss << merge_scope.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
  }

  return ss.str();
}

std::string TPipe::LocalTQueAlloc() const {
  stringstream ss;

  for (auto& [id, que] : this->ques) {
    stringstream tensor_size_max;
    stringstream tensor_depth_max;
    stringstream tensor_bufnum_max;

    tensor_size_max << KernelUtils::Max() << "(";
    tensor_depth_max << KernelUtils::Max() << "(";
    tensor_bufnum_max << KernelUtils::Max() << "(";

    bool is_first = true;

    for (auto mid : que.merge_scopes) {
      auto merge_scope = this->merge_scopes.find(mid);
      if (merge_scope == this->merge_scopes.end()) {
        throw std::runtime_error("Merge scope not found: " + std::to_string(mid));
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
        tensor_depth_max << ", ";
        tensor_bufnum_max << ", ";
      }

      tensor_size_max << merge_scope->second.size;
      tensor_depth_max << merge_scope->second.depth;
      tensor_bufnum_max << merge_scope->second.buf_num;
    }

    for (auto tid : que.not_merge_tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        throw std::runtime_error("Tensor not found: " + std::to_string(tid));
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
        tensor_depth_max << ", ";
        tensor_bufnum_max << ", ";
      }

      tensor_size_max << tensor->second.size << " * sizeof(" << Tensor::DtypeName(tensor->second.dtype) << ")";
      tensor_depth_max << tensor->second.que_depth;
      tensor_bufnum_max << tensor->second.que_buf_num;
    }

    tensor_size_max << ")";
    tensor_depth_max <<")";
    tensor_bufnum_max << ")";

    ss << que.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << que.depth.DefineConst(tensor_depth_max.str()) << std::endl;
    ss << que.buf_num.DefineConst(tensor_bufnum_max.str()) << std::endl;
    ss << que.Define() << std::endl;
    ss << this->InitTQueBuffers(que) << std::endl;
    ss << std::endl;
  }

  return ss.str();
}

std::string TPipe::LocalTBufAlloc() const {
  stringstream ss;

  for (auto &[id, buf] : this->bufs) {
    stringstream tensor_size_max;
    tensor_size_max << KernelUtils::Max() << "(";

    bool is_first = true;
    for (auto mid : buf.merge_scopes) {
      auto merge_scope = this->merge_scopes.find(mid);
      if (merge_scope == this->merge_scopes.end()) {
        throw std::runtime_error("Merge scope not found: " + std::to_string(mid));
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
      }

      tensor_size_max << merge_scope->second.size;
    }

    for (auto tid : buf.not_merge_tensors) {
      auto tensor = this->tensors.find(tid);
      if (tensor == this->tensors.end()) {
        throw std::runtime_error("Tensor not found: " + std::to_string(tid));
      }

      if (is_first) {
        is_first = false;
      } else {
        tensor_size_max << ", ";
      }

      tensor_size_max << tensor->second.size << " * sizeof(" << Tensor::DtypeName(tensor->second.dtype) << ")";
    }

    tensor_size_max << ")";

    ss << buf.size.DefineConst(tensor_size_max.str()) << std::endl;
    ss << buf.Define() << std::endl;
    ss << this->InitTBufBuffer(buf) << std::endl;
    ss << buf.AllocBuf() << std::endl;
    ss << std::endl;
  }

  return ss.str();
}

std::string TPipe::LocalTensorQueBufAlloc() const {
  stringstream ss;

  ss << this->TensorSizeCalc();
  ss << std::endl;
  ss << this->MergeScopeSizeCalc();
  ss << std::endl;
  ss << this->LocalTBufAlloc();
  ss << this->LocalTQueAlloc();

  return ss.str();
}

int Looper::LoopAxisDistance(const std::vector<ascir::AxisId> &axis, const ascir::AxisId loop_axis) {
  if (axis.size() == 0 || loop_axis == ascir::ID_NONE) {
    return -this->current_axis.size();
  }

  int same_axis_num = 0;
  for (int i = 0; i < axis.size() && i < this->current_axis.size(); ++i) {
    if (axis[i] == this->current_axis[i]) {
      same_axis_num++;
    } else if (axis[i] == loop_axis) {
      break;
    }
  }

  int loop_axis_pos = -1;
  for (int i = 0; i < axis.size(); ++i) {
    if (loop_axis == axis[i]) {
      loop_axis_pos = i;
      break;
    }
  }

  if (loop_axis_pos < 0) {
    throw std::runtime_error("Node loop axis not found in axis.");
  }

  if (same_axis_num < this->current_axis.size()) {
    if (loop_axis_pos < same_axis_num) {
      return - (this->current_axis.size() - loop_axis_pos);
    } else {
      return - (this->current_axis.size() - same_axis_num);
    }
  } else {
    return (loop_axis_pos + 1) - this->current_axis.size();
  }
}

ApiCall::ApiCall(const ascir::NodeView &node) {
  this->type = node->GetType();

  for (auto input : node.inputs()) {
    if (input == nullptr) {
      continue;
    }

    this->inputs.emplace_back(input->mem.tensor_id);
  }

  for (auto output : node.outputs()) {
    if (output == nullptr) {
      continue;
    }

    this->outputs.emplace_back(output.mem.tensor_id);
  }
}

std::string NopApicall(std::string func, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) {
    return "";
}

std::string CastApicall(std::string unaryname, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                        const std::vector<std::reference_wrapper<const Tensor>> &outputs) {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  stringstream ss;
  ss << "Cast" << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
     << "RoundMode::CAST_NONE, " << x.size << ");" << std::endl;
  return ss.str();
}

std::string UnaryApicall(std::string unaryname, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                         const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                         const std::vector<std::reference_wrapper<const Tensor>> &outputs) {
    auto x = inputs[0].get();
    auto y = outputs[0].get();
    stringstream ss;
    ss << unaryname << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << x.size << ");" << std::endl;
    return ss.str();
}

std::string BinaryApicall(std::string binaryname, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                          const std::vector<std::reference_wrapper<const Tensor>> &outputs) {
    auto x1 = inputs[0].get();
    auto x2 = inputs[1].get();
    auto y = outputs[0].get();
    stringstream ss;
    ss << binaryname << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
       << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
       << x1.size << ");" << std::endl;
    return ss.str();
}

std::string ReduceApicall(std::string api_name, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                          const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                          const std::vector<std::reference_wrapper<const Tensor>> &outputs) {
    std::map<std::string, pair<std::string, std::string>> api_names = {
        {"Max", {"WholeReduceMax", "Max"}},
        {"Min", {"WholeReduceMin", "Min"}},
        {"Sum", {"WholeReduceSumAdapt", "Add"}},
    };
    auto iter = api_names.find(api_name);
    if (iter == api_names.end()) {
        throw std::runtime_error("Unsupported reduce api: " + api_name);
    }

    auto& [reduce_func, binary_func] = iter->second;

    auto x = inputs[0].get();
    auto y = outputs[0].get();

    auto m_pos = x.vectorized_axis_pos[0];
    auto k_pos = x.vectorized_axis_pos[1];

    stringstream ss;
    ss << "ReduceLast<"
            << Tensor::DtypeName(y.dtype) << ", "
            << reduce_func << ", "
            << binary_func
        << ">" << "("
        << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
        << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
        << tpipe.tiler.Size(x.axis_size[m_pos]) << ", "
        << tpipe.tiler.Size(x.axis_size[k_pos])
        << ");" << std::endl;
    return ss.str();
}

std::vector<uint32_t> NonZeroVecAxisPos(const Tensor& tensor) {
  std::vector<uint32_t> axis_pos;
  for (auto pos : tensor.vectorized_axis_pos) {
    if (tensor.axis_strides[pos] == 0 || tensor.axis_size[pos] == 0) {
        continue;
    }
    axis_pos.push_back(pos);
  }
  return axis_pos;
}

std::string LoadApiCall(const std::string &api_name, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                        const std::vector<std::reference_wrapper<const Tensor>> &outputs)
{
    std::stringstream ss;
    auto gm = inputs[0].get();
    auto ub = outputs[0].get();

    auto ub_non0_pos = NonZeroVecAxisPos(ub);
    if ((ub_non0_pos.size() == 1 && ub.axis_strides[ub_non0_pos[0]] == 1)
        || (ub_non0_pos.size() == 2 && ub.axis_strides[ub_non0_pos[0]] == ub.axis_size[ub_non0_pos[1]])) {
      ss << "DataCopyPad("
          << ub << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, ub) << "], "
          << gm << "[" << tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides) << "], "
          << "DataCopyParams("
            << "1, "
            << "(uint16_t)(" << ub.size << " * sizeof(" <<  Tensor::DtypeName(ub.dtype) << ")), "
            << "0, "
            << "0"
          << "), "
          << "DataCopyPadParams()"
          << ");" << std::endl;
    } else if (ub_non0_pos.size() == 2 && ub.axis_strides[ub_non0_pos[1]] == 1) {
      // Set last axis as stride axis
      auto out = ub.vectorized_axis_pos[0];
      auto in = ub.vectorized_axis_pos[1];
      auto& tiler = tpipe.tiler;
      ss << "DataCopy("
          << ub << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, ub) << "], "
          << gm << "[" << tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides) << "], "
          << "DataCopyParams("
            << tiler.Size(ub.axis_size[out]) << ", "
            << tiler.BlkLen(ub.dtype, ub.axis_size[in]) << ", "
            << tiler.BlkLenDiff(ub.dtype, ub.axis_strides[out], ub.axis_size[in]) << ", "
            << "0" // Assert UB is continously, only decide src stride
          << ")"
          << ");" << std::endl;
    } else {
      throw std::runtime_error("Not support load api call.");
    }
    return ss.str();
}

std::string StoreApiCall(const std::string &api_name, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                         const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                         const std::vector<std::reference_wrapper<const Tensor>> &outputs)
{
    std::stringstream ss;
    auto gm = outputs[0].get();
    auto ub = inputs[0].get();

    auto ub_non0_pos = NonZeroVecAxisPos(ub);
    if ((ub_non0_pos.size() == 1 && ub.axis_strides[ub_non0_pos[0]] == 1)
        || (ub_non0_pos.size() == 2 && ub.axis_strides[ub_non0_pos[0]] == ub.axis_size[ub_non0_pos[1]])) {
      ss << "DataCopyPad("
          << gm << "[" << tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides) << "], "
          << ub << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, ub) << "], "
          << "DataCopyParams("
            << "1, "
            << "(uint16_t)(" << ub.size << " * sizeof(" <<  Tensor::DtypeName(ub.dtype) << ")), "
            << "0, "
            << "0"
          << ")"
          << ");" << std::endl;
    } else if (ub_non0_pos.size() == 2 && ub.axis_strides[ub_non0_pos[1]] == 1) {
      // Set last axis as stride axis
      auto out = ub.vectorized_axis_pos[0];
      auto in = ub.vectorized_axis_pos[1];
      auto& tiler = tpipe.tiler;
      ss << "DataCopy("
          << gm << "[" << tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides) << "], "
          << ub << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, ub) << "], "
          << "DataCopyParams("
            << tiler.Size(ub.axis_size[out]) << ", "
            << tiler.BlkLen(ub.dtype, ub.axis_size[in]) << ", "
            << "0, " // Assert UB is continously, only decide dst stride
            << tiler.BlkLenDiff(ub.dtype, ub.axis_strides[out], ub.axis_size[in])
          << ")"
          << ");" << std::endl;
    } else {
      throw std::runtime_error("Not support store api call.");
    }
    return ss.str();
}

std::string BroadcastApiCall(const std::string& api_name, const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                             const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                             const std::vector<std::reference_wrapper<const Tensor>> &outputs)
{
    auto x = inputs[0].get();
    auto y = outputs[0].get();
    if (x.vectorized_axis.size() != 2 || y.vectorized_axis.size() != 2 || x.vectorized_axis != y.vectorized_axis) {
      throw std::runtime_error("Broadcast only support 2D vectorized axis.");
    }

    auto& src_m_size = x.axis_size[x.vectorized_axis_pos[0]];
    auto& src_k_size = x.axis_size[x.vectorized_axis_pos[1]];
    auto& dst_m_size = y.axis_size[y.vectorized_axis_pos[0]];
    auto& dst_k_size = y.axis_size[y.vectorized_axis_pos[1]];

    std::stringstream ss;
    ss << "BroadCastLastImpl("
       << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
       << x << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
       << "BroadCastLastND{"
         << tpipe.tiler.Size(dst_m_size) << ", " << tpipe.tiler.Size(dst_k_size) << ", "
         << tpipe.tiler.Size(src_m_size) << ", " << tpipe.tiler.Size(src_k_size)
         << "}"
       << ");" << std::endl;
    return ss.str();
}

std::string ApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  using Funcptr = std::function<std::string(std::string, const TPipe&, const std::vector<ascir::AxisId>&,
                                            const std::vector<std::reference_wrapper<const Tensor>>&,
                                            const std::vector<std::reference_wrapper<const Tensor>>&)>;
  map<std::string, pair<std::string, Funcptr>> type2apicall = {
      {Load::Type, {"Load", LoadApiCall}},
      {Store::Type, {"Store", StoreApiCall}},
      {Broadcast::Type, {"Broadcast", BroadcastApiCall}},
      {Nop::Type, {"Nop", NopApicall}},
      {Cast::Type, {"Cast", CastApicall}},
      {Abs::Type, {"Abs", UnaryApicall}},
      {Exp::Type, {"Exp", UnaryApicall}},
      {Div::Type, {"Div", BinaryApicall}},
      {Sub::Type, {"Sub", BinaryApicall}},
      {Max::Type, {"Max", ReduceApicall}},
      {Sum::Type, {"Sum", ReduceApicall}},
  };

  stringstream ss;
  auto it = type2apicall.find(this->type);
  if (it == type2apicall.end()) {
      throw std::runtime_error("Unsupported API call: " + this->type);
  }

  auto &[api_name, func] = it->second;
  ss << func(api_name, tpipe, current_axis, inputs, outputs);
  return ss.str();
}

Stage::Stage(ascir::ComputeUnit unit) : unit(unit) {}

void Stage::AddCall(const ascir::NodeView &node) {
  if (node.attr.api.unit != this->unit) {
    throw std::runtime_error("Stage add different unit call.");
  }

  for (auto input : node.inputs()) {
    if (input == nullptr) {
      continue;
    }

    this->reads.insert(input->mem.tensor_id);
    if (input->mem.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
      this->read_ques.insert(input->que.id);
    }
  }

  for (auto output : node.outputs()) {
    if (output == nullptr) {
      continue;
    }

    this->writes.insert(output.mem.tensor_id);
    if (output.mem.alloc_type == ascir::ALLOC_TYPE_QUEUE) {
      this->write_ques.insert(output.que.id);
    }
  }

  this->calls.emplace_back(ApiCall(node));
}

std::string Stage::GenerateApiCalls(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis) const
{
  std::stringstream ss;

  std::set<ascir::TensorId> alloced_tensors;
  std::set<ascir::TensorId> unsynced_tensors;

  // api call
  for (auto call : this->calls) {
    for (auto i: call.inputs) {
      if (unsynced_tensors.find(i) != unsynced_tensors.end()) {
          ss << "pipe_barrier(PIPE_V);" << std::endl;
          unsynced_tensors.clear();
          break;
      }
    }

    std::vector<reference_wrapper<const Tensor>> input_tensors;
    std::vector<reference_wrapper<const Tensor>> output_tensors;
    for (auto &[ids, tensors] : {pair{call.inputs, &input_tensors}, {call.outputs, &output_tensors}}) {
      for (auto id : ids) {
        auto& tensor = tpipe.GetTensor(id);
        tensors->emplace_back(ref(tensor));

        if (tensor.alloc_type != ascir::ALLOC_TYPE_GLOBAL && alloced_tensors.find(id) == alloced_tensors.end()) {
          alloced_tensors.insert(id);
          ss << tpipe.TensorAlloc(tensor);
        }
      }
    }

    ss << call.Generate(tpipe, current_axis, input_tensors, output_tensors);

    for (auto o: call.outputs) {
      unsynced_tensors.insert(o);
    }

    // Not need to free tensor, all tensor will be free when Enque/Free buffer
  }

  return ss.str();
}

std::string Stage::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId>& current_axis) const {
  stringstream ss;
  ss << "{" << std::endl;

  // Read/Write que/buf DeQue/alloc
  for (auto read_que_id : this->read_ques) {
    auto que = tpipe.GetQue(read_que_id);
    ss << que.DequeBuf() << std::endl;
  }
  for (auto write_que_id : this->write_ques) {
    auto que = tpipe.GetQue(write_que_id);
    ss << que.AllocBuf() << std::endl;
  }

  ss << this->GenerateApiCalls(tpipe, current_axis);

  // Read/Write que/buf Enque/free
  for (auto write_que_id : this->write_ques) {
    auto que = tpipe.GetQue(write_que_id);
    ss << que.EnqueBuf() << std::endl;
  }
  for (auto read_que_id : this->read_ques) {
    auto que = tpipe.GetQue(read_que_id);
    ss << que.FreeBuf() << std::endl;
  }

  ss << "}" << std::endl;
  return ss.str();
}

Loop::Loop(ascir::AxisId axis) : axis(axis) {}

void Loop::AddLoop(codegen::Loop::LoopId loop) {
    this->body.emplace_back(Loop::LOOP, loop);
}

void Loop::AddStage(codegen::Loop::StageId stage) {
    this->body.emplace_back(Loop::STAGE, stage);
}

void Looper::EnterLoop(ascir::AxisId axis) {
    if (this->current_loops.empty()) {
        throw std::runtime_error("Not root loop to enter");
    }

    if (this->current_stage != ascir::ID_NONE) {
        this->ExitStage();
    }

    auto current_loop_id = current_loops.back();
    Loop::LoopId new_loop_id = this->loops.size();
    this->loops.emplace_back(Loop{axis});

    this->loops[current_loop_id].AddLoop(new_loop_id);
    this->current_loops.emplace_back(new_loop_id);
    this->current_axis.emplace_back(axis);
}

void Looper::ExitLoop() {
    if (this->current_loops.size() == 1) {
        throw std::runtime_error("Can not exit root loop");
    }

    if (this->current_stage != ascir::ID_NONE) {
        ExitStage();
    }

    this->current_loops.pop_back();
    this->current_axis.pop_back();
}

void Looper::InitRootLoop() {
    this->current_stage = ascir::ID_NONE;
    this->current_axis.clear();
    this->current_loops.clear();

    this->root_loop = this->loops.size();
    this->loops.emplace_back(Loop(ascir::ID_NONE));
    this->current_loops.emplace_back(this->root_loop);
}

void Looper::EnterStage(ascir::ComputeUnit unit) {
    if (this->current_stage != ascir::ID_NONE) {
        throw std::runtime_error("already in stage");
    }

    if (this->current_loops.empty()) {
        throw std::runtime_error("Not loop to add stage.");
    }

    auto new_stage_id = this->stages.size();
    this->stages.emplace_back(Stage{unit});

    this->loops[this->current_loops.back()].AddStage(new_stage_id);
    this->current_stage = new_stage_id;
};

void Looper::ExitStage() {
    if (this->current_stage == ascir::ID_NONE) {
        throw std::runtime_error("Not in Stage");
    }
    this->current_stage = ascir::ID_NONE;
}

void Looper::EndRootLoop() {
    this->current_stage = ascir::ID_NONE;
    this->current_axis.clear();
    this->current_loops.clear();
}

void Looper::AddNode(const ascir::NodeView node) {
  auto axis = node.attr.sched.axis();
  auto loop_axis = node.attr.sched.loop_axis();

  int loop_distance = this->LoopAxisDistance(axis, loop_axis);
  while (loop_distance != 0) {
    if (loop_distance > 0) {
        this->EnterLoop(axis[this->current_axis.size()]);
    } else {
        this->ExitLoop();
    }

    loop_distance = this->LoopAxisDistance(axis, loop_axis);
  }

  auto unit = node.attr.api.unit;
  if (this->current_stage == ascir::ID_NONE) {
    this->EnterStage(unit);
  } else {
    if (this->stages[current_stage].unit != unit) {
        this->ExitStage();
        this->EnterStage(unit);
    } else {
        // Pass in same unit
    }
  }

  auto& stage = this->stages[this->current_stage];

  stage.AddCall(node);
}

void Looper::GenerateLoop(const Tiler &tiler, const TPipe& tpipe, std::vector<ascir::AxisId>& current_axis, const Loop &loop, std::stringstream &ss) const {
  for (auto& [type, body] : loop.body) {
    if (type == Loop::LOOP) {
        auto loop = this->loops[body];
        current_axis.push_back(loop.axis);

        auto axis = tiler.GetAxis(loop.axis);
        if (axis.type == Axis::AXIS_TYPE_BLOCK_OUTER) {
            this->GenerateLoop(tiler, tpipe, current_axis, loop, ss);
        } else {
            std::string loop_size = axis.type == ascir::Axis::AXIS_TYPE_BLOCK_INNER ? axis.loop_size.name : tiler.Size(axis.size);
            ss << "for (" << axis.AsArg() << " = 0; " << axis << " < " << loop_size << "; " << axis << "++) {" << std::endl;
            this->GenerateLoop(tiler, tpipe,  current_axis, loop, ss);
            ss << "}" << std::endl;
        }

        current_axis.pop_back();
    } else if (type == Loop::STAGE) {
        auto stage = this->stages[body];
        ss << stage.Generate(tpipe, current_axis);
    } else {
        throw std::runtime_error("Unknown body type.");
    }
  }
}

std::string Looper::GenerateLoop(const Tiler &tiler, const TPipe& tpipe) const {
  stringstream ss;
  std::vector<ascir::AxisId> current_axis;
  this->GenerateLoop(tiler, tpipe, current_axis, this->loops[this->root_loop], ss);
  return ss.str();
}

Kernel::Kernel()
    : tiling_data_arg("tiling"),
      workspace_arg("workspace"),
      tpipe("tpipe", this->tiler) {}

std::string Kernel::IncludeAndDefines() const {
    std::stringstream ss;
    ss << "#ifdef __CCE_KT_TEST__" << std::endl;
    ss << "#include \"tikicpulib.h\"" << std::endl;
    ss << "#include \"" << CamelToLowerSneak(this->name) << "_tiling.h\"" << std::endl;
    ss << "#define GET_TILING_DATA(tiling_data, tiling) \\" << std::endl;
    ss << "    optiling::TilingData& tiling_data = *(optiling::TilingData*)(tiling);" << std::endl;
    ss << "#endif" << std::endl;
    ss << std::endl;
    ss << "#include \"kernel_operator.h\"" << std::endl;
    ss << std::endl;
    ss << "using namespace AscendC;" << std::endl;
    return ss.str();
}

std::string Kernel::KernelFunctionDeclare() const {
    const char* flags[] = {"extern \"C\"", "__global__", "__aicore__"};
    const char* return_type = "void";

    std::stringstream ss;
    for (auto flag : flags) {
      ss << flag << " ";
    }
    ss << return_type << " ";
    ss << CamelToLowerSneak(this->name) << "(";
    for (auto& input : this->inputs) {
      ss << input.AsArg() << ", ";
    }
    for (auto& output : this->outputs) {
      ss << output.AsArg() << ", ";
    }
    ss << this->workspace_arg.AsArg() << ", ";
    ss << this->tiling_data_arg.AsArg();
    ss << ")";
    return ss.str();
}

std::string Kernel::GlobalTensorInit() const {
  std::stringstream ss;
  for (int i = 0; i < this->inputs.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->input_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      throw std::runtime_error("Input tensor not found: " + std::to_string(this->input_tensors[i]));
    }

    ss << tensor->second.Define() << std::endl;
    ss << tensor->second.SetGlobalBuffer(this->inputs[i]) << std::endl;
  }

  for (int i = 0; i < this->outputs.size(); i++) {
    auto tensor = this->tpipe.tensors.find(this->output_tensors[i]);
    if (tensor == this->tpipe.tensors.end()) {
      throw std::runtime_error("Output tensor not found: " + std::to_string(this->output_tensors[i]));
    }

    ss << tensor->second.Define() << std::endl;
    ss << tensor->second.SetGlobalBuffer(this->outputs[i]) << std::endl;
  }

  return ss.str();
}

std::string Kernel::LocalTensorQueBufAlloc() const {
  stringstream ss;

  ss << this->tpipe.Define() << std::endl;
  ss << std::endl;

  ss << this->tpipe.LocalTensorQueBufAlloc();

  ss << std::endl;

  return ss.str();
}

std::string Kernel::TmpBufAlloc() const {
  stringstream ss;
  ss << "TBuf<TPosition::VECCALC> tmp;" << std::endl
   << "tpipe.InitBuffer(tmp, 8 * 1024);" << std::endl
   << "LocalTensor<uint8_t> tmp_buf = tmp.Get<uint8_t>();" << std::endl
   << "SetTmp(tmp_buf);" << std::endl;
  return ss.str();
}

Kernel Kernel::ParseGraph(const ascir::ImplGraph &graph) {
  Kernel kernel;

  kernel.name = graph.GetName();

  // Parse kernel input output
  for (auto input : graph.GraphInputs()) {
    if (!IsOps<Data>(input)) {
      throw std::runtime_error("Unsupported input type: " + input->GetName() + " " + input->GetType());
    }
    kernel.inputs.emplace_back(GM_ADDR(input->GetName()));
    kernel.input_tensors.emplace_back(input.outputs[0].mem.tensor_id);
  }

  for (auto output : graph.GraphOutputs()) {
    if (!IsOps<Output>(output)) {
      throw std::runtime_error("Unsupported output type: " + output->GetName() + " " + output->GetType());
    }
    kernel.outputs.emplace_back(GM_ADDR(output->GetName()));
    kernel.output_tensors.emplace_back(output.inputs[0]->mem.tensor_id);
  }

  // Parse for tiler
  for (auto axis : graph.axis()) {
    kernel.tiler.AddAxis(axis);
  }

  for (auto size : graph.size_var()) {
    kernel.tiler.AddSizeVar(size);
  }

  // Parse for tpipe
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Output>(node)) {
      continue;
    }

    auto desc = node->GetOpDesc();
    for (auto output : node.outputs()) {
      auto tensor_name = node->GetName() + "_" + desc->GetOutputNameByIndex(output.Index());
      kernel.tpipe.AddTensor(output, tensor_name);
    }
  }

  // Parse for loop
  kernel.looper.InitRootLoop();
  for (auto node : graph.GetAllNodes()) {
    if (IsOps<Data>(node) || IsOps<Output>(node)) {
      continue;
    }

    kernel.looper.AddNode(node);
  }
  kernel.looper.EndRootLoop();

  return kernel;
}

std::string Kernel::Generate() {
  stringstream ss;

  ss << this->IncludeAndDefines();
  ss << std::endl;

  ss << KernelUtils::FunctionDefines();
  ss << std::endl;

  ss << AscendCApiExtend::FunctionDefines();
  ss << std::endl;

  ss << this->KernelFunctionDeclare() <<  " {" << std::endl;

  ss << this->tiler.TilingDataDefine(this->tiling_data_arg);
  ss << std::endl;
  ss << this->tiler.BlockOutterAxisDefine();
  ss << std::endl;
  ss << this->tiler.LoopSizeCal();
  ss << std::endl;
  ss << this->GlobalTensorInit();
  ss << std::endl;
  ss << this->LocalTensorQueBufAlloc();

  ss << this->TmpBufAlloc();

  ss << this->looper.GenerateLoop(this->tiler, this->tpipe);

  ss << "}" << std::endl;

  return ss.str();
}
