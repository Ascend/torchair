#ifndef __CODEGEN_KERNEL_H__
#define __CODEGEN_KERNEL_H__

#include "ascir.h"

namespace codegen {
class Code {
 public:
  virtual std::string Str() const = 0;
};

class Type : public Code {
 public:
  const std::string name;

  explicit Type(const std::string& name);
  std::string Str() const override;
};

const Type Void_t{"void"};
const Type Int_t{"int"};
const Type Int32_t{"int32_t"};
const Type Int64_t{"int64_t"};
const Type Uint32_t{"uint32_t"};
const Type Half_t{"half"};
const Type GM_ADDR_t{"GM_ADDR"};

class Variable : public Code {
public:
  const Type type;
  const std::string name;

  Variable(const Type &type, const std::string &name);
  std::string Str() const override;
  std::string AsArg() const;
  std::string Define(std::string &&init = "", bool define_const = false) const;
  inline std::string DefineConst(std::string &&init = "") const {
    return Define(std::move(init), true);
  }
};

struct Int : public Variable {
  explicit inline Int(const std::string &name) : Variable(Int_t, name) {}
};

struct GM_ADDR : public Variable {
  explicit inline GM_ADDR(std::string name) : Variable{GM_ADDR_t, name} {};
};

struct Uint32 : public Variable {
  explicit inline Uint32(const std::string &name) : Variable(Uint32_t, name) {}
};

class Axis : public ascir::Axis, public Code {
 public:
  Int var;

  explicit Axis(const ascir::Axis &axis);
  std::string Str() const override;
};

class Tensor : public Variable {
 public:
  static const std::string& DtypeName(ge::DataType dtype);
  static const Type& GlobalTensorTypes(ge::DataType dtype);
  static const Type& LocalTensorTypes(ge::DataType dtype);

  ascir::TensorId id;
  ge::DataType dtype;
  ascir::AllocType alloc_type;
  ascir::Position position;

  vector<ascir::AxisId> axis;
  vector<ascir::AxisId> vectorized_axis;
  vector<ascir::SizeExpr> axis_size;
  vector<ascir::SizeExpr> axis_strides;

  ascir::QueId que_id;
  ascir::BufId buf_id;

  Uint32 size; /** Que/Buf size in element number */
  Uint32 que_depth;
  Uint32 que_buf_num;

  uint32_t que_depth_value;
  uint32_t que_buf_num_value;

  ascir::MergeScopeId merge_scope;

  explicit Tensor(const ascir::TensorAttr& tensor, const std::string& name = "");

  // For GlobalTensor
  std::string SetGlobalBuffer(GM_ADDR global) const;
};

std::string PositionValue(ascir::Position position);

class MergeScope {
 public:
  ascir::MergeScopeId id;
  ascir::Position position;
  std::vector<ascir::TensorId> tensors;

  Uint32 size;
  Uint32 depth;
  Uint32 buf_num;

  MergeScope(ascir::MergeScopeId id, ascir::Position position);
};

class TQue : public Variable {
 public:
  ascir::QueId id;
  ascir::Position position;
  std::set<ascir::MergeScopeId> merge_scopes;
  std::set<ascir::TensorId> not_merge_tensors;

  Uint32 size;
  Uint32 depth;
  Uint32 buf_num;

  TQue(ascir::QueId id, ascir::Position position);
};

class TBuf : public Variable {
 public:
  ascir::BufId id;
  ascir::Position position;
  std::set<ascir::MergeScopeId> merge_scopes;
  std::set<ascir::TensorId> not_merge_tensors;

  Uint32 size;

  TBuf(ascir::BufId id, const ascir::Position position);
};

class Tiler : public Code {
 public:
  Variable tiling_data;

  Int block_dim;
  std::map<ascir::AxisId, codegen::Axis> axis;
  std::map<ascir::SizeVarId, ascir::SizeVar> sizes;

  explicit Tiler(const std::string &tiling_data_name = "t");

  void AddSizeVar(const ascir::SizeVar &size);
  void AddAxis(const ascir::Axis &axis);

  std::string Str() const override;
  std::string Size(const ascir::SizeExpr& size) const;
  std::string TensorVectorizedSize(const Tensor& tensor) const;
  const Axis& GetAxis(const ascir::AxisId id) const;
  std::string AxisSize(const ascir::AxisId id) const;
  std::string AxisSize(const Axis& axis) const;
  std::string AxisName(const ascir::AxisId axis_id) const;

  /** 根据tiling参数GM地址获取tiling_data */
  std::string TilingDataDefine(GM_ADDR tiling_data_arg) const;

  /**
   * 定义Block外轴，并用blockidx初始化。
   */
  std::string BlockOutterAxisDefine();
};

class TPipe : public Variable {
 public:
  const Tiler& tiler;

  map<ascir::TensorId, Tensor> tensors;
  map<ascir::MergeScopeId, MergeScope> merge_scopes;
  map<ascir::QueId, TQue> ques;
  map<ascir::BufId, TBuf> bufs;

  TPipe(const std::string &name, const Tiler &tiler);
  Tensor& AddTensor(const Tensor &tensor);
  Tensor& AddTensor(const ascir::TensorAttr &tensor, const std::string& name = "");

  std::string InitTQueBuffers(const TQue &que) const;
  std::string InitTBufBuffer(const TBuf &buf) const;

  std::string TensorSizeCalc() const;
  std::string MergeScopeSizeCalc() const;
  std::string LocalTBufAlloc() const;
  std::string LocalTQueAlloc() const;
  std::string LocalTensorQueBufAlloc() const;
};

class KernelUtils {
 public:
  static std::string FunctionDefines();
  static std::string Max();
  static std::string Sum();
};

class Kernel {
 public:
  GM_ADDR workspace_arg;
  GM_ADDR tiling_data_arg;
  std::vector<GM_ADDR> inputs;
  std::vector<ascir::TensorId> input_tensors;
  std::vector<GM_ADDR> outputs;
  std::vector<ascir::TensorId> output_tensors;

  std::string name;

  Tiler tiler;
  TPipe tpipe;

  Kernel();

  static Kernel ParseGraph(const ascir::ImplGraph &graph);
  std::string Generate();

  std::string IncludeAndDefines() const;
  std::string KernelFunctionDeclare() const;
  std::string GlobalTensorInit() const;
  std::string LocalTensorQueBufAlloc() const;
};
}

std::ostream &operator<<(std::ostream &os, const codegen::Code &obj);

#endif
