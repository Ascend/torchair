#include <sstream>

#include "ascir_ops.h"
#include "codegen_kernel.h"

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
    : ascir::Axis(axis), var(axis.name) {}

std::string Axis::Str() const {
  return name;
}

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
      merge_scope(tensor.opt.merge_scope) {}

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
      buf_num(this->name + "_buf_num") {}

TBuf::TBuf(ascir::BufId id, const ascir::Position position)
    : Variable(Type("TBuf<" + PositionValue(position) + ">"), "b" + to_string(id)),
      id(id),
      position(position),
      size(this->name + "_size") {}

Tiler::Tiler(const std::string &tiling_data_name)
    : tiling_data(Type{"optiling::TilingData"}, tiling_data_name),
      block_dim("block_dim")
{}

std::string Tiler::Str() const {
  return tiling_data.Str();
}

void codegen::Tiler::AddSizeVar(const ascir::SizeVar &size) {
  this->sizes.emplace(size.id, size);
}

void codegen::Tiler::AddAxis(const ascir::Axis &axis) {
  this->axis.emplace(axis.id, codegen::Axis(axis));
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

  for (auto axis : tensor.vectorized_axis) {
    auto iter = std::find(tensor.axis.begin(), tensor.axis.end(), axis);
    if (iter == tensor.axis.end()) {
      throw std::runtime_error("Vectorized axis " + to_string(axis) + " not found in tensor " + tensor.name);
    }

    if (first) {
      first = false;
    } else {
      ss << " + ";
    }

    auto index = iter - tensor.axis.begin();
    ss << "(" + this->Size(tensor.axis_size[index]) + " - 1)";
    if (!(tensor.axis_strides[index] == 1)) {
        ss << " * " << this->Size(tensor.axis_strides[index]);
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

std::string codegen::Tiler::BlockOutterAxisDefine() {
  stringstream code;

  code << this->block_dim.Define("GetBlockIdx()") << std::endl;

  for (auto& [id, axis] : this->axis) {
    if (axis.type != ascir::Axis::AXIS_TYPE_BLOCK_OUTER) {
      continue;
    }

    stringstream axis_value;
    axis_value << this->block_dim.name << " % (" << this->AxisSize(axis) << ")";
    code << axis.var.Define(axis_value.str(), true);
    code << " ";

    code << this->block_dim.name << " /= " << this->AxisSize(axis) << ";";
    code << std::endl;
  }

  return code.str();
}

std::string KernelUtils::FunctionDefines() {
  std::stringstream ss;
  ss << "namespace utils {" << std::endl;
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
  ss << "}" << std::endl;

  return ss.str();
}

std::string KernelUtils::Max() {
  return "utils::Max";
}

std::string KernelUtils::Sum() {
  return "utils::Sum";
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
    auto iter = this->bufs.find(t.id);
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

std::string TPipe::InitTQueBuffers(const TQue &que) const {
  stringstream ss;
  ss << this->name << "."
     << "InitBuffer(" << que << ", " << que.buf_num << ", " << que.size << ");";
  return ss.str();
}

std::string TPipe::InitTBufBuffer(const TBuf &buf) const {
  stringstream ss;
  ss << this->name << "."
     << "InitBuffer(" << buf << ", " << buf.size << ");";
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

Kernel::Kernel()
    : tiling_data_arg("tiling"),
      workspace_arg("workspace"),
      tpipe("tpipe", this->tiler) {}

std::string Kernel::IncludeAndDefines() const {
    std::stringstream ss;
    ss << "#ifdef __CCE_KT_TEST__" << std::endl;
    ss << "#include \"tikicpulib.h\"" << std::endl;
    ss << "#include \"load_abs_store_tiling.h\"" << std::endl;
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
    ss << this->name << "(";
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

  return kernel;
}

std::string Kernel::Generate() {
  stringstream ss;

  ss << this->IncludeAndDefines();
  ss << std::endl;

  ss << KernelUtils::FunctionDefines();
  ss << std::endl;

  ss << this->KernelFunctionDeclare() <<  " {" << std::endl;

  ss << this->tiler.TilingDataDefine(this->tiling_data_arg);
  ss << std::endl;
  ss << this->tiler.BlockOutterAxisDefine();
  ss << std::endl;
  ss << this->GlobalTensorInit();
  ss << std::endl;
  ss << this->LocalTensorQueBufAlloc();

  ss << "}" << std::endl;

  return ss.str();
}
