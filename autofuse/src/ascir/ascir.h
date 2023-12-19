#ifndef __ASCIR_H__
#define __ASCIR_H__

#include <vector>
#include "graph/graph.h"

#include "compute_graph.h"
#include "graph_utils_ex.h"
#include "op_desc_utils.h"

namespace ascir {
using EnumValue = int32_t;

using Position = EnumValue;
enum : EnumValue {
  POSITION_GM,
  POSITION_VECIN,
  POSITION_VECOUT,
};

using AllocType = EnumValue;
enum : EnumValue { ALLOC_TYPE_GLOBAL, ALLOC_TYPE_BUFFER, ALLOC_TYPE_QUEUE };

using MemHardware = EnumValue;
enum : EnumValue {
  MEM_HARDWARE_GM,
  MEM_HARDWARE_UB,
};

// Start: As extend to AscendC Api
using ComputeUnit = EnumValue;
enum : EnumValue {
  UNIT_NONE,
  UNIT_MTE,
  UNIT_SCALAR,
  UNIT_VECTOR,
  UNIT_CUBE,
};

using ApiType = EnumValue;
enum : EnumValue {
  API_TYPE_BUFFER,
  API_TYPE_COMPUTE
};

// Start: Ascir it's self needs
using Identifier = int64_t;
enum : Identifier {
  ID_NONE = -1,
};

using SizeVarId = Identifier;
struct SizeVar {
  using Type = EnumValue;
  enum : EnumValue {
    SIZE_TYPE_VAR = 0,
    SIZE_TYPE_CONST = 1,
  };
  static const char *TypeStr(Type type);

  SizeVarId id;
  std::string name;

  Type type;
  int64_t value; /**< for SIZE_TYPE_CONST */
};

struct SizeExpr {
  std::vector<SizeVarId> nums;
  std::vector<SizeVarId> dens;

  SizeExpr operator/(const SizeExpr &rhs);
  SizeExpr &operator/=(const SizeExpr &rhs);
  SizeExpr operator*(const SizeExpr &rhs);
  SizeExpr &operator*=(const SizeExpr &rhs);
  bool operator==(const SizeExpr &rhs) const;

  SizeExpr(std::initializer_list<SizeVarId> nums = {}, std::initializer_list<SizeVarId> dens = {});
};

using AxisId = Identifier;
struct Axis {
  using Type = EnumValue;
  enum : EnumValue {
    AXIS_TYPE_ORIGINAL,
    AXIS_TYPE_BLOCK_OUTER,
    AXIS_TYPE_BLOCK_INNER,
    AXIS_TYPE_TILE_OUTER,
    AXIS_TYPE_TILE_INNER,
    AXIS_TYPE_MERGED
  };
  static const char *TypeStr(Type type);

  AxisId id;
  std::string name;

  Type type;
  SizeExpr size;
  std::vector<AxisId> from;
};

template <typename Holder, const char *ATTR_NAME, typename FieldType>
struct AttrField;

template <typename Holder, const char *ATTR_NAME>
struct AttrField<Holder, ATTR_NAME, int32_t> {
  Holder holder;

 public:
  void operator=(const int32_t &other) {
    ge::AttrUtils::SetInt(holder, ATTR_NAME, other);
  }

  int32_t operator()() const {
    int32_t value = 0;
    ge::AttrUtils::GetInt(holder, ATTR_NAME, value);
    return value;
  }

  operator int32_t() const {
    return this->operator()();
  };
};

template <typename Holder, const char *ATTR_NAME>
struct AttrField<Holder, ATTR_NAME, int64_t> {
  Holder holder;

 public:
  void operator=(const int64_t &other) {
    ge::AttrUtils::SetInt(holder, ATTR_NAME, other);
  }

  int64_t operator()() const {
    int64_t value = 0;
    ge::AttrUtils::GetInt(holder, ATTR_NAME, value);
    return value;
  }

  operator int64_t() const {
    return this->operator()();
  };
};

template <typename Holder, const char *ATTR_NAME>
struct AttrField<Holder, ATTR_NAME, vector<Identifier>> {
  Holder holder;

 public:
  void operator=(const vector<Identifier> &other) {
    ge::AttrUtils::SetListInt(holder, ATTR_NAME, other);
  }

  Identifier operator[](int index) const {
    vector<Identifier> values;
    ge::AttrUtils::GetListInt(holder, ATTR_NAME, values);
    return values[index];
  }

  vector<Identifier> operator()() const {
    vector<Identifier> value;
    ge::AttrUtils::GetListInt(holder, ATTR_NAME, value);
    return value;
  };

  operator vector<Identifier>() const {
    return this->operator()();
  };
};

template <typename Holder, const char *ATTR_NAME_PREFIX>
struct AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>> {
  Holder holder;

 public:
  static const std::string NUM_OF_FACTOR;
  static const std::string NUMS;
  static const std::string DENS;
  void operator=(const vector<SizeExpr> &values) {
    ge::AttrUtils::SetInt(holder, NUM_OF_FACTOR, values.size());

    std::vector<std::vector<ascir::SizeVarId>> nums;
    std::vector<std::vector<ascir::SizeVarId>> dens;
    for (auto expr : values) {
      nums.push_back(expr.nums);
      dens.push_back(expr.dens);
    }

    ge::AttrUtils::SetListListInt(holder, NUMS.c_str(), nums);
    ge::AttrUtils::SetListListInt(holder, DENS.c_str(), dens);
  }

  vector<SizeExpr> operator()() const {
    std::vector<std::vector<ascir::SizeVarId>> nums;
    std::vector<std::vector<ascir::SizeVarId>> dens;
    ge::AttrUtils::GetListListInt(holder, NUMS.c_str(), nums);
    ge::AttrUtils::GetListListInt(holder, DENS.c_str(), dens);

    std::vector<ascir::SizeExpr> result;
    result.reserve(nums.size());
    for (size_t i = 0; i < nums.size(); ++i) {
      ascir::SizeExpr expr;
      expr.nums = nums[i];
      expr.dens = dens[i];
      result.push_back(expr);
    }
    return result;
  };

  SizeExpr operator[](int index) const {
    int64_t num = 0;
    ge::AttrUtils::GetInt(holder, NUM_OF_FACTOR.c_str(), num);
    if (index >= num) {
      throw std::out_of_range("index out of range");
    }
    return this->operator()()[index];
  }
};

template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::NUM_OF_FACTOR{
    std::string(ATTR_NAME_PREFIX) + ".num_of_factor"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::NUMS{
    std::string(ATTR_NAME_PREFIX) + ".nums"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::DENS{
    std::string(ATTR_NAME_PREFIX) + ".dens"};

template <typename Holder, const char *ATTR_NAME_PREFIX>
struct AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeVar>> {
  Holder holder;

 public:
  static const std::string NUM;
  static const std::string NAME;
  static const std::string TYPE;
  static const std::string VALUE;

  void operator=(std::initializer_list<ascir::SizeVar> &&size_vars) {
    ge::AttrUtils::SetInt(holder, NUM, size_vars.size());

    std::vector<std::string> names;
    std::vector<SizeVar::Type> types;
    std::vector<int64_t> values;
    for (auto var : size_vars) {
      names.push_back(var.name);
      types.push_back(var.type);
      values.push_back(var.value);
    }

    ge::AttrUtils::SetListStr(holder, NAME, names);
    ge::AttrUtils::SetListInt(holder, TYPE, types);
    ge::AttrUtils::SetListInt(holder, VALUE, values);
  }

  vector<SizeVar> operator()() const {
    std::vector<std::string> names;
    std::vector<SizeVar::Type> types;
    std::vector<int64_t> values;
    ge::AttrUtils::GetListStr(holder, NAME, names);

    ge::AttrUtils::GetListInt(holder, TYPE, types);
    ge::AttrUtils::GetListInt(holder, VALUE, values);

    std::vector<ascir::SizeVar> result;
    result.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      ascir::SizeVar var;
      var.id = i;
      var.name = names[i];
      var.type = types[i];
      var.value = values[i];
      result.push_back(var);
    }
    return result;
  }

  SizeVar operator[](int index) const {
    int64_t num = 0;
    ge::AttrUtils::GetInt(holder, NUM.c_str(), num);
    if (index >= num) {
      throw std::out_of_range("index out of range");
    }

    return this->operator()()[index];
  }
};

template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeVar>>::NUM{std::string(ATTR_NAME_PREFIX) +
                                                                                        ".num"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeVar>>::NAME{std::string(ATTR_NAME_PREFIX) +
                                                                                         ".name"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeVar>>::TYPE{std::string(ATTR_NAME_PREFIX) +
                                                                                         ".type"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::SizeVar>>::VALUE{
    std::string(ATTR_NAME_PREFIX) + ".value"};

template <typename Holder, const char *ATTR_NAME_PREFIX>
struct AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>> {
  Holder holder;

 public:
  static const std::string NUM;
  static const std::string NAME;
  static const std::string TYPE;
  static const std::string SIZE_NUMS;
  static const std::string SIZE_DENS;
  static const std::string FROM;

  void operator=(std::initializer_list<ascir::Axis> &&values) {
    ge::AttrUtils::SetInt(holder, NUM.c_str(), values.size());

    std::vector<std::string> names;
    std::vector<Axis::Type> types;
    std::vector<vector<SizeVarId>> size_nums;
    std::vector<vector<SizeVarId>> size_dens;
    std::vector<vector<AxisId>> from;

    for (auto axis : values) {
      names.push_back(axis.name);
      types.push_back(axis.type);
      size_nums.push_back(axis.size.nums);
      size_dens.push_back(axis.size.dens);
      from.push_back(axis.from);
    }

    ge::AttrUtils::SetListStr(holder, NAME, names);
    ge::AttrUtils::SetListInt(holder, TYPE, types);
    ge::AttrUtils::SetListListInt(holder, SIZE_NUMS, size_nums);
    ge::AttrUtils::SetListListInt(holder, SIZE_DENS, size_dens);
    ge::AttrUtils::SetListListInt(holder, FROM, from);
  }

  vector<Axis> operator()() const {
    std::vector<string> names;
    std::vector<Axis::Type> types;
    std::vector<vector<SizeVarId>> size_nums;
    std::vector<vector<SizeVarId>> size_dens;
    std::vector<vector<AxisId>> from;

    ge::AttrUtils::GetListStr(holder, NAME, names);
    ge::AttrUtils::GetListInt(holder, TYPE, types);
    ge::AttrUtils::GetListListInt(holder, SIZE_NUMS, size_nums);
    ge::AttrUtils::GetListListInt(holder, SIZE_DENS, size_dens);
    ge::AttrUtils::GetListListInt(holder, FROM, from);

    std::vector<ascir::Axis> result;
    result.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      ascir::Axis axis;
      axis.id = i;
      axis.name = names[i];
      axis.type = types[i];
      axis.size.nums = size_nums[i];
      axis.size.dens = size_dens[i];
      axis.from = from[i];
      result.push_back(axis);
    }

    return result;
  }

  Axis operator[](int index) const {
    int64_t num = 0;
    ge::AttrUtils::GetInt(holder, NUM.c_str(), num);
    if (index >= num) {
      throw std::out_of_range("index out of range");
    }

    return this->operator()()[index];
  }
};

template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::NUM{std::string(ATTR_NAME_PREFIX) +
                                                                                     ".num"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::NAME{std::string(ATTR_NAME_PREFIX) +
                                                                                      ".name"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::TYPE{std::string(ATTR_NAME_PREFIX) +
                                                                                      ".type"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::SIZE_NUMS{
    std::string(ATTR_NAME_PREFIX) + ".size_nums"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::SIZE_DENS{
    std::string(ATTR_NAME_PREFIX) + ".size_dens"};
template <typename Holder, const char *ATTR_NAME_PREFIX>
const std::string AttrField<Holder, ATTR_NAME_PREFIX, std::vector<ascir::Axis>>::FROM{std::string(ATTR_NAME_PREFIX) +
                                                                                      ".from"};

using TensorId = Identifier;
using BufId = Identifier;
using QueId = Identifier;
using MergeScopeId = Identifier;

class TensorAttr {
 public:
  template <const char *ATTR_NAME, typename T>
  using Fields = AttrField<ge::GeTensorDesc *, ATTR_NAME, T>;

  static constexpr char AXIS[] = "ascir.tensor.axis";
  static constexpr char REPEATS[] = "ascir.tensor.repeats";
  static constexpr char STRIDES[] = "ascir.tensor.strides";
  static constexpr char VECTORIZED_AXIS[] = "ascir.tensor.vectorized_axis";

  static constexpr char MEM_TENSOR_ID[] = "ascir.tensor.mem.tensor_id";
  static constexpr char MEM_ALLOC_TYPE[] = "ascir.tensor.mem.alloc_type";
  static constexpr char MEM_HARDWARE[] = "ascir.tensor.mem.hardware";
  static constexpr char MEM_POSITION[] = "ascir.tensor.mem.position";

  static constexpr char QUE_ID[] = "ascir.tensor.que.id";
  static constexpr char QUE_DEPTH[] = "ascir.tensor.que.depth";
  static constexpr char QUE_BUF_NUM[] = "ascir.tensor.que.buf_num";

  static constexpr char BUF_ID[] = "ascir.tensor.buf.id";

  static constexpr char OPT_REF_TENSOR[] = "ascir.tensor.opt.ref_tensor";
  static constexpr char OPT_MERGE_SCOPE[] = "ascir.tensor.opt.merge_scope";

  union {
    ge::GeTensorDesc *desc;
    Fields<AXIS, vector<AxisId>> axis;
    Fields<REPEATS, vector<SizeExpr>> repeats;
    Fields<STRIDES, vector<SizeExpr>> strides;
    Fields<VECTORIZED_AXIS, vector<AxisId>> vectorized_axis;

    union {
      Fields<MEM_TENSOR_ID, TensorId> tensor_id;
      Fields<MEM_ALLOC_TYPE, AllocType> alloc_type;
      Fields<MEM_HARDWARE, MemHardware> hardware;
      Fields<MEM_POSITION, Position> position;
    } mem;

    union {
      Fields<QUE_ID, QueId> id;
      Fields<QUE_DEPTH, int64_t> depth;
      Fields<QUE_BUF_NUM, int64_t> buf_num;
    } que;

    union {
      Fields<BUF_ID, BufId> id;
    } buf;

    union {
      Fields<OPT_REF_TENSOR, TensorId> ref_tensor;
      Fields<OPT_MERGE_SCOPE, MergeScopeId> merge_scope;
    } opt;
  };

  explicit inline TensorAttr(ge::GeTensorDesc *desc) : desc(desc) {}
};

/**
 * Tensor are follow the output
 */
template <const int32_t OP_OUTPUT_INDEX, const char *ATTR_NAME, typename T>
class OutputAttrField {
  ge::Operator *op;

 public:
  void operator=(const T &other) {
    op->SetOutputAttr(OP_OUTPUT_INDEX, ATTR_NAME, other);
  }

  operator T() const {
    T value;
    op->GetOutputAttr(OP_OUTPUT_INDEX, ATTR_NAME, value);
    return value;
  };
};

template <int OUTPUT_INDEX, const char *ATTR_NAME_PREFIX>
struct OutputAttrField<OUTPUT_INDEX, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>> {
  ge::Operator *op;

 public:
  static const std::string NUM_OF_FACTOR;
  static const std::string NUMS;
  static const std::string DENS;
  void operator=(std::initializer_list<ascir::SizeExpr> &&values) {
    op->SetOutputAttr(OUTPUT_INDEX, NUM_OF_FACTOR.c_str(), (int64_t)values.size());

    std::vector<std::vector<ascir::SizeVarId>> nums;
    std::vector<std::vector<ascir::SizeVarId>> dens;
    for (auto expr : values) {
      nums.push_back(expr.nums);
      dens.push_back(expr.dens);
    }

    auto opdesc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    ge::AttrUtils::SetListListInt(opdesc->MutableOutputDesc(0), NUMS.c_str(), nums);
    ge::AttrUtils::SetListListInt(opdesc->MutableOutputDesc(0), DENS.c_str(), dens);
  }
};

template <int OUTPUT_INDEX, const char *ATTR_NAME_PREFIX>
const std::string OutputAttrField<OUTPUT_INDEX, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::NUM_OF_FACTOR{
    std::string(ATTR_NAME_PREFIX) + ".num_of_factor"};
template <int OUTPUT_INDEX, const char *ATTR_NAME_PREFIX>
const std::string OutputAttrField<OUTPUT_INDEX, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::NUMS{
    std::string(ATTR_NAME_PREFIX) + ".nums"};
template <int OUTPUT_INDEX, const char *ATTR_NAME_PREFIX>
const std::string OutputAttrField<OUTPUT_INDEX, ATTR_NAME_PREFIX, std::vector<ascir::SizeExpr>>::DENS{
    std::string(ATTR_NAME_PREFIX) + ".dens"};

template <int OUTPUT_INDEX>
struct OutputAttrDataType {
  ge::Operator *op;

 public:
  void operator=(const ge::DataType &other) {
    auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    desc->MutableOutputDesc(OUTPUT_INDEX)->SetDataType(other);
  }

  operator ge::DataType() const {
    auto desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
    return desc->GetOutputDesc(OUTPUT_INDEX).GetDataType();
  };
};

template <int OUTPUT_INDEX>
union OperatorOutput {
  ge::Operator *__op;
  OutputAttrDataType<OUTPUT_INDEX> dtype;

  static constexpr char AXIS[] = "ascir.tensor.axis";
  OutputAttrField<OUTPUT_INDEX, AXIS, std::vector<AxisId>> axis;

  static constexpr char REPEATS[] = "ascir.tensor.repeats";
  OutputAttrField<OUTPUT_INDEX, REPEATS, std::vector<SizeExpr>> repeats;

  static constexpr char STRIDES[] = "ascir.tensor.strides";
  OutputAttrField<OUTPUT_INDEX, STRIDES, std::vector<SizeExpr>> strides;
};

template <int INPUT_INDEX>
struct OperatorInput {
 protected:
  ge::Operator *op;

 public:
  template <const int OUTPUT_INDEX>
  void operator=(const OperatorOutput<OUTPUT_INDEX> &output) {
    this->op->SetInput((uint32_t)INPUT_INDEX, *output.__op, OUTPUT_INDEX);
  }

  void operator=(const ge::Operator &op) {
    this->op->SetInput((uint32_t)INPUT_INDEX, op, 0);
  }
};

struct NodeAttr {
  static constexpr char SCHED_EXEC_ORDER[] = "ascir.op.sched.exec_order";
  static constexpr char SCHED_AXIS[] = "ascir.op.sched.axis";
  static constexpr char SCHED_LOOP_AXIS[] = "ascir.op.sched.loop_axis";

  static constexpr char API_UNIT[] = "ascir.op.api.unit";
  static constexpr char API_TYPE[] = "ascir.op.api.type";
  static constexpr char API_INPUT_POSITIONS[] = "ascir.op.api.input_postions";
  static constexpr char API_OUTPUT_POSITIONS[] = "ascir.op.api.output_positions";

  union {
    ge::OpDesc *__opdesc;
    union {
      AttrField<ge::OpDesc *, SCHED_EXEC_ORDER, int64_t> exec_order;
      AttrField<ge::OpDesc *, SCHED_AXIS, vector<AxisId>> axis;
      AttrField<ge::OpDesc *, SCHED_LOOP_AXIS, AxisId> loop_axis;
    } sched;

    union {
      AttrField<ge::OpDesc *, API_TYPE, ApiType> type;
      AttrField<ge::OpDesc *, API_UNIT, ComputeUnit> unit;
    } api;

    union {
    } hint;

    union {
    } impl;
  };

  explicit inline NodeAttr(ge::NodePtr node) {
    this->__opdesc = node->GetOpDescBarePtr();
  }
  explicit inline NodeAttr(ge::Operator& op) {
    this->__opdesc = ge::OpDescUtils::GetOpDescFromOperator(op).get();
  }
};

struct TensorView : public TensorAttr, public ge::OutDataAnchorPtr {
  inline ge::NodePtr Owner() {
    if (this == nullptr || this->get() == nullptr) {
      return nullptr;
    }
    return this->get()->GetOwnerNode();
  }

  inline const uint32_t Index() {
    if (this->get() == nullptr) {
      return -1;
    }

    return this->get()->GetIdx();
  };

  explicit inline TensorView(ge::OutDataAnchorPtr anchor)
      : ge::OutDataAnchorPtr(anchor),
        TensorAttr(anchor == nullptr
                       ? nullptr
                       : anchor->GetOwnerNode()->GetOpDescBarePtr()->MutableOutputDesc(anchor->GetIdx()).get()) {}
};

struct TensorPtr : public ge::InDataAnchorPtr {
  TensorView _view;

  explicit TensorPtr(ge::InDataAnchorPtr anchor) : ge::InDataAnchorPtr(anchor), _view(nullptr) {}

  TensorView *operator->() {
    auto peer = this->get()->GetPeerOutAnchor();
    if (peer == nullptr) {
      return nullptr;
    }

    this->_view = TensorView(peer);
    return &_view;
  }
};

inline bool operator==(const TensorView &lhs, const TensorPtr &rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    throw std::runtime_error("TensorView and TensorPtr can not be null");
  }

  return rhs.get()->GetPeerOutAnchor().get() == lhs.get();
}

inline bool operator==(const TensorPtr &lhs, const TensorView &rhs) {
  if (lhs == nullptr || rhs == nullptr) {
    throw std::runtime_error("TensorView and TensorPtr can not be null");
  }

  return lhs.get()->GetPeerOutAnchor().get() == rhs.get();
}

struct InputsView {
  ge::NodePtr _node;

  explicit inline InputsView(ge::NodePtr node) : _node(node) {}

  inline std::vector<TensorPtr> operator()() {
    std::vector<TensorPtr> views;
    for (int i = 0; i < _node->GetAllInDataAnchorsSize(); ++i) {
      views.push_back(TensorPtr{_node->GetInDataAnchor(i)});
    }

    return views;
  };

  inline TensorPtr operator[](int index) const {
    if (index >= _node->GetAllInDataAnchorsSize()) {
      throw std::out_of_range(std::to_string(index));
    }

    return TensorPtr{_node->GetInDataAnchor(index)};
  };
};

struct OutputsView {
  ge::NodePtr _node;

  explicit inline OutputsView(ge::NodePtr node) : _node(node) {}

  inline std::vector<TensorView> operator()() {
    std::vector<TensorView> views;
    for (int i = 0; i < _node->GetAllOutDataAnchorsSize(); ++i) {
      views.push_back(TensorView{_node->GetOutDataAnchor(i)});
    }

    return views;
  };

  inline TensorView operator[](int index) {
    if (index >= _node->GetAllOutDataAnchorsSize()) {
      throw std::out_of_range(std::to_string(index));
    }

    return TensorView{_node->GetOutDataAnchor(index)};
  }
};

struct NodeView : public ge::NodePtr {
  NodeAttr attr;
  InputsView inputs;
  OutputsView outputs;

  explicit inline NodeView(ge::NodePtr node) : ge::NodePtr(node), attr(node), inputs(node), outputs(node){};
};

class NodeViewIter : protected ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator {
 public:
  explicit NodeViewIter(ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator &&iter);
  NodeViewIter &operator++();
  NodeView operator*();
  bool operator!=(const NodeViewIter &other) const;
};

class NodeViewVisitor : protected ge::ComputeGraph::Vistor<ge::NodePtr> {
 public:
  using Iterator = ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator;
  NodeViewIter begin();
  NodeViewIter end();

  NodeViewVisitor();
  explicit NodeViewVisitor(ge::ComputeGraph::Vistor<ge::NodePtr> &&visitor);
};

class Graph : public ge::Graph {
 public:
  static constexpr char SIZE_VAR[] = "ascir.graph.size_var";
  static constexpr char AXIS[] = "ascir.graph.axis";

  union {
    ge::AttrHolder *holder;
    AttrField<ge::AttrHolder *, SIZE_VAR, std::vector<SizeVar>> size_var;
    AttrField<ge::AttrHolder *, AXIS, std::vector<Axis>> axis;
  };

  /* Every time setting the input will build new graph,
   * we need to trace it for copy the axis/size_var table
   */
  Graph &SetInputs(const std::vector<ge::Operator> &inputs);
  using ge::Graph::SetOutputs;

  explicit Graph(const char *name);

  SizeVar CreateSizeVar(const std::string &name, SizeVar::Type type, int64_t value);
  SizeVar CreateSizeVar(const std::string &name);
  SizeVar CreateSizeVar(const std::string &name, const int64_t value);

  /**
   * Construct a new ORIGINAL axis.
   * @detail
   *   This will automatically increase the number of axis_id.
   *   And set the "from" fields to empty sets;
   */
  Axis CreateAxis(const std::string &name, Axis::Type type, const SizeExpr &size, const std::vector<AxisId> &from);
  Axis CreateAxis(const std::string &name, const SizeExpr &size);

  std::tuple<Axis, Axis> BlockSplit(AxisId axis);
  std::tuple<Axis, Axis> TileSplit(AxisId axis);
  Axis MergeAxis(std::initializer_list<AxisId> axis);

  void ApplySplit(NodeView &node, AxisId outter, AxisId inner, AxisId original);
  void ApplyMerge(NodeView &node, AxisId new_axis, std::initializer_list<AxisId> original);

  NodeView Find(const char *name);
  NodeViewVisitor GetAllNodes();

  int CopyFrom(ascir::Graph& graph);

 private:
  ge::NamedAttrs tmp_attr_holder;
};

using HintGraph = Graph;
using ImplGraph = Graph;
} // namespace ascir

// Operator register
#include "graph/operator_reg.h"
#define REG_OPS(type) \
  namespace ge {      \
  REG_OP(type)

#define END_OPS(type)      \
  OP_END_FACTORY_REG(type) \
  }                        \
  ;

#endif