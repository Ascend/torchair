#include "ascir.h"
#include <stdexcept>
#include <string>

#include "graph_utils_ex.h"
#include "graph_utils.h"
#include "node_utils_ex.h"

using namespace ascir;

const char* SizeVar::TypeStr(Type type) {
  constexpr static const char *TypeToStr[] = {
      [SIZE_TYPE_VAR] = "VAR",
      [SIZE_TYPE_CONST] = "CONST"
  };

  if (type >= sizeof(TypeToStr) / sizeof(TypeToStr[0])) {
    return "";
  }
  return TypeToStr[type];
}

SizeExpr::SizeExpr(std::initializer_list<SizeVar> nums, std::initializer_list<SizeVar> dens) {
  for (auto &num : nums) {
    this->nums.push_back(num.id);
  }

  for (auto den : dens) {
    this->dens.push_back(den.id);
  }

  std::sort(this->nums.begin(), this->nums.end());
  std::sort(this->dens.begin(), this->dens.end());
}

SizeExpr::SizeExpr(std::initializer_list<SizeVarId> nums, std::initializer_list<SizeVarId> dens) {
  this->nums = nums;
  this->dens = dens;
  std::sort(this->nums.begin(), this->nums.end());
  std::sort(this->dens.begin(), this->dens.end());
}

SizeExpr SizeExpr::operator/(const SizeExpr &rhs) const {
  SizeExpr result = *this;
  result.nums.insert(result.nums.end(), rhs.dens.begin(), rhs.dens.end());
  result.dens.insert(result.dens.end(), rhs.nums.begin(), rhs.nums.end());
  std::sort(result.nums.begin(), result.nums.end());
  std::sort(result.dens.begin(), result.dens.end());

  if (rhs.is_zero) {
    throw std::invalid_argument("division by zero");
  }
  return result;
}

SizeExpr &SizeExpr::operator/=(const SizeExpr &rhs) {
  this->nums.insert(this->nums.end(), rhs.dens.begin(), rhs.dens.end());
  this->dens.insert(this->dens.end(), rhs.nums.begin(), rhs.nums.end());
  std::sort(this->nums.begin(), this->nums.end());
  std::sort(this->dens.begin(), this->dens.end());

  if (rhs.is_zero) {
    throw std::invalid_argument("division by zero");
  }
  return *this;
}

SizeExpr SizeExpr::operator*(const SizeExpr &rhs) const {
  SizeExpr result = *this;
  result.is_zero = this->is_zero || rhs.is_zero;
  result.nums.insert(result.nums.end(), rhs.nums.begin(), rhs.nums.end());
  result.dens.insert(result.dens.end(), rhs.dens.begin(), rhs.dens.end());
  std::sort(result.nums.begin(), result.nums.end());
  std::sort(result.dens.begin(), result.dens.end());
  return result;
}

SizeExpr &SizeExpr::operator*=(const SizeExpr &rhs) {
  this->is_zero = this->is_zero || rhs.is_zero;
  this->nums.insert(this->nums.end(), rhs.nums.begin(), rhs.nums.end());
  this->dens.insert(this->dens.end(), rhs.dens.begin(), rhs.dens.end());
  std::sort(this->nums.begin(), this->nums.end());
  std::sort(this->dens.begin(), this->dens.end());
  return *this;
}

bool SizeExpr::operator==(const SizeExpr &rhs) const {
  if (rhs.is_zero && this->is_zero) {
    return true;
  }

  if (!std::is_sorted(this->nums.begin(), this->nums.end()) ||
      !std::is_sorted(this->dens.begin(), this->dens.end()) ||
      !std::is_sorted(rhs.nums.begin(), rhs.nums.end()) ||
      !std::is_sorted(rhs.dens.begin(), rhs.dens.end())) {
    throw std::runtime_error("SizeExpr not sorted");
  }

  return this->nums == rhs.nums && this->dens == rhs.dens;
}

bool SizeExpr::operator==(const int64_t rhs) const {
  if (rhs == 0 && this->is_zero) {
    return true;
  }

  if (rhs == 1 && this->nums.size() == 0 && this->dens.size() == 0 && this->is_zero == false) {
    return true;
  }

  return false;
}

bool SizeExpr::operator!=(const SizeExpr &rhs) const {
  if (rhs.is_zero != this->is_zero) {
    return true;
  }

  if (!std::is_sorted(this->nums.begin(), this->nums.end()) ||
      !std::is_sorted(this->dens.begin(), this->dens.end()) ||
      !std::is_sorted(rhs.nums.begin(), rhs.nums.end()) ||
      !std::is_sorted(rhs.dens.begin(), rhs.dens.end())) {
    throw std::runtime_error("SizeExpr not sorted");
  }

  return this->nums != rhs.nums || this->dens != rhs.dens;
}

const char *Axis::TypeStr(Type type) {
  static const char *TypeToStr[] = {
      [AXIS_TYPE_ORIGINAL] = "ORIGINAL",
      [AXIS_TYPE_BLOCK_OUTER] = "BLOCK_OUTER",
      [AXIS_TYPE_BLOCK_INNER] = "BLOCK_INNER",
      [AXIS_TYPE_TILE_OUTER] = "TILE_OUTER",
      [AXIS_TYPE_TILE_INNER] = "TILE_INNER",
      [AXIS_TYPE_MERGED] = "MERGED"
  };
  if (type >= sizeof(TypeToStr) / sizeof(TypeToStr[0])) {
    return "";
  }
  return TypeToStr[type];
}

Graph::Graph(const char *name) : ge::Graph(name) {
  // 当前没有直接设置本Graph的ComputeGraph指针的能力，通过构造一个新的加Copy来绕路完成该能力
  auto compute_graph = std::make_shared<ge::ComputeGraph>(name);
  auto tmp_graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  ge::GraphUtilsEx::CopyGraph(tmp_graph, *this);

  holder = ge::GraphUtilsEx::GetComputeGraph(*this).get();
}


Graph &Graph::SetInputs(const std::vector<ge::Operator> &inputs) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);

  // Will build new compute_graph after set inputs
  ge::Graph::SetInputs(inputs);
  auto new_compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (new_compute_graph == nullptr) {
    return *this;
  }

  new_compute_graph->CopyAttrsFrom(*this->holder);

  this->holder = new_compute_graph.get();
  return *this;
}

SizeVar Graph::CreateSizeVar(const std::string &name, SizeVar::Type type, int64_t value) {
  int64_t num_of_size_var = 0;
  ge::AttrUtils::GetInt(this->holder, this->size_var.NUM, num_of_size_var);

  SizeVar new_size;
  new_size.id = num_of_size_var;
  new_size.name = name;
  new_size.type = type;
  new_size.value = value;

  vector<string> size_names;
  vector<Axis::Type> size_types;
  vector<int64_t> size_values;

  ge::AttrUtils::GetListStr(this->holder, this->size_var.NAME, size_names);
  ge::AttrUtils::GetListInt(this->holder, this->size_var.TYPE, size_types);
  ge::AttrUtils::GetListInt(this->holder, this->size_var.VALUE, size_values);

  size_names.push_back(name);
  size_types.push_back(type);
  size_values.push_back(value);

  ge::AttrUtils::SetListStr(this->holder, this->size_var.NAME, size_names);
  ge::AttrUtils::SetListInt(this->holder, this->size_var.TYPE, size_types);
  ge::AttrUtils::SetListInt(this->holder, this->size_var.VALUE, size_values);

  ge::AttrUtils::SetInt(this->holder, this->size_var.NUM, num_of_size_var + 1);

  return new_size;
}

SizeVar Graph::CreateSizeVar(const std::string &name) {
  return Graph::CreateSizeVar(name, SizeVar::SIZE_TYPE_VAR, 0);
}

SizeVar Graph::CreateSizeVar(const std::string &name, const int64_t value) {
  return Graph::CreateSizeVar(name, SizeVar::SIZE_TYPE_CONST, value);
}

Axis Graph::CreateAxis(const std::string &name, Axis::Type type, const SizeExpr &size, const vector<AxisId> &from) {
  int64_t num_of_axis = 0;
  ge::AttrUtils::GetInt(this->holder, this->axis.NUM, num_of_axis);

  Axis new_axis;
  new_axis.id = num_of_axis;
  new_axis.type = type;
  new_axis.name = name;
  new_axis.size = size;
  new_axis.from = from;
  new_axis.align = 1;
  new_axis.allow_oversize_axis = false;
  new_axis.allow_unaligned_tail = true;

  vector<string> axis_names;
  vector<Axis::Type> axis_types;
  vector<vector<SizeVarId>> axis_size_num;
  vector<vector<SizeVarId>> axis_size_den;
  vector<vector<AxisId>> axis_from;
  vector<int32_t> aligns;
  vector<bool> allow_oversize_axis;
  vector<bool> allow_unaligned_tail;

  ge::AttrUtils::GetListStr(this->holder, this->axis.NAME, axis_names);
  ge::AttrUtils::GetListInt(this->holder, this->axis.TYPE, axis_types);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.SIZE_NUMS, axis_size_num);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.SIZE_DENS, axis_size_den);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.FROM, axis_from);
  ge::AttrUtils::GetListInt(this->holder, this->axis.ALIGN, aligns);
  ge::AttrUtils::GetListBool(this->holder, this->axis.ALLOW_OVERSIZE_AXIS, allow_oversize_axis);
  ge::AttrUtils::GetListBool(this->holder, this->axis.ALLOW_UNALIGNED_TAIL, allow_unaligned_tail);

  axis_names.push_back(name);
  axis_types.push_back(type);
  axis_size_num.push_back(size.nums);
  axis_size_den.push_back(size.dens);
  axis_from.push_back(from);
  aligns.push_back(new_axis.align);
  allow_oversize_axis.push_back(new_axis.allow_oversize_axis);
  allow_unaligned_tail.push_back(new_axis.allow_unaligned_tail);

  ge::AttrUtils::SetListStr(this->holder, this->axis.NAME, axis_names);
  ge::AttrUtils::SetListInt(this->holder, this->axis.TYPE, axis_types);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.SIZE_NUMS, axis_size_num);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.SIZE_DENS, axis_size_den);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.FROM, axis_from);
  ge::AttrUtils::SetListInt(this->holder, this->axis.ALIGN, aligns);
  ge::AttrUtils::SetListBool(this->holder, this->axis.ALLOW_OVERSIZE_AXIS, allow_oversize_axis);
  ge::AttrUtils::SetListBool(this->holder, this->axis.ALLOW_UNALIGNED_TAIL, allow_unaligned_tail);

  ge::AttrUtils::SetInt(this->holder, this->axis.NUM, num_of_axis + 1);
  return new_axis;
}

Axis Graph::CreateAxis(const std::string &name, const SizeExpr &size) {
  return CreateAxis(name, Axis::AXIS_TYPE_ORIGINAL, size, {});
}

std::tuple<Axis, Axis> Graph::BlockSplit(AxisId axis_id, std::string innerDimName, std::string outterDimName) {
  auto axis = this->axis[axis_id];
  if (innerDimName.empty()) {
    innerDimName = axis.name + "b";
  }
  if (outterDimName.empty()) {
    outterDimName = axis.name + "B";
  }
  auto block_size = this->CreateSizeVar(innerDimName + "_size", SizeVar::SIZE_TYPE_VAR, 0);

  auto outter = this->CreateAxis(outterDimName, Axis::AXIS_TYPE_BLOCK_OUTER,
                                 axis.size / SizeExpr{{block_size.id}}, {axis_id});
  auto inner = this->CreateAxis(innerDimName,
                                Axis::AXIS_TYPE_BLOCK_INNER,
                                SizeExpr{{block_size.id}}, {axis_id});
  return std::make_tuple(outter, inner);
}

std::tuple<Axis, Axis> Graph::TileSplit(AxisId axis_id, std::string innerDimName, std::string outterDimName) {
  auto axis = this->axis[axis_id];
  if (innerDimName.empty()) {
    innerDimName = axis.name + "t";
  }
  if (outterDimName.empty()) {
    outterDimName = axis.name + "T";
  }
  auto tile_size = this->CreateSizeVar(innerDimName + "_size", SizeVar::SIZE_TYPE_VAR, 0);

  auto outter = this->CreateAxis(outterDimName,
                                 Axis::AXIS_TYPE_TILE_OUTER,
                                 axis.size / SizeExpr{{tile_size.id}}, {axis_id});
  auto inner = this->CreateAxis(innerDimName,
                                Axis::AXIS_TYPE_TILE_INNER,
                                SizeExpr{{tile_size.id}}, {axis_id});
  return std::make_tuple(outter, inner);
}

Axis Graph::MergeAxis(const std::vector<AxisId>& axis) {
  string name;
  SizeExpr size;
  vector<AxisId> from;

  for (auto &i : axis) {
    name += this->axis[i].name;
    size *= this->axis[i].size;
    from.push_back(i);
  }

  return this->CreateAxis(name, Axis::AXIS_TYPE_MERGED, size, from);
}

void Graph::ApplySplit(NodeView &node, AxisId outter_id, AxisId inner_id, AxisId original) {
  SizeExpr split_size{this->axis[inner_id].size};

  vector<AxisId> new_axis;
  auto axis = node.attr.sched.axis();
  for (auto &i : axis) {
    if (i == original) {
      new_axis.push_back(outter_id);
      new_axis.push_back(inner_id);
    } else {
      new_axis.push_back(i);
    }
  }

  node.attr.sched.axis = new_axis;

  bool found = false;
  for (int i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    vector<AxisId> new_axis;
    vector<SizeExpr> new_repeat;
    vector<SizeExpr> new_strides;

    auto const axis = node.outputs[i].axis();
    auto const repeat = node.outputs[i].repeats();
    auto const strides = node.outputs[i].strides();

    for (int a = 0; a < axis.size(); a++) {
      if (axis[a] != original) {
        new_axis.push_back(axis[a]);
        new_repeat.push_back(repeat[a]);
        new_strides.push_back(strides[a]);
      } else {
        found = true;
        new_axis.push_back(outter_id);
        new_axis.push_back(inner_id);

        if (repeat[a] == 1 && strides[a] == 0) {
          // broadcast axis
          new_repeat.push_back(SizeExpr::One());
          new_strides.push_back(SizeExpr::Zero());

          new_repeat.push_back(SizeExpr::One());
          new_strides.push_back(SizeExpr::Zero());
        } else {
          new_repeat.push_back(repeat[a] / split_size);
          new_strides.push_back(strides[a] * split_size);

          new_repeat.push_back(split_size);
          new_strides.push_back(strides[a]);
        }
      }
    }

    node.outputs[i].axis = new_axis;
    node.outputs[i].repeats = new_repeat;
    node.outputs[i].strides = new_strides;
  }

  return;
}

void Graph::ApplySplit(NodeView &node, AxisId outter_id, AxisId inner_id) {
  if (outter_id >= this->axis.Size() || inner_id >= this->axis.Size()) {
    throw std::runtime_error("outter_id or inner_id is out of range.");
  }

  auto out_axis = this->axis[outter_id];
  auto in_axis = this->axis[inner_id];
  if (!((out_axis.type == Axis::AXIS_TYPE_BLOCK_OUTER && in_axis.type == Axis::AXIS_TYPE_BLOCK_INNER) ||
        (out_axis.type == Axis::AXIS_TYPE_TILE_OUTER && in_axis.type == Axis::AXIS_TYPE_TILE_INNER))) {
    throw std::runtime_error("Outter and inner axis must be all block split or all tile split axis.");
  }

  if (out_axis.from.size() != 1 || in_axis.from.size()!= 1 || out_axis.from[0] != in_axis.from[0]) {
    throw std::runtime_error("Outter and inner axis must be split from same axis.");
  }

  ApplySplit(node, outter_id, inner_id, out_axis.from[0]);
}

void Graph::ApplyMerge(NodeView &node, AxisId merged_axis_id, const std::vector<AxisId> &original) {
  vector<AxisId> new_axis;

  auto cur = original.begin();
  for (auto i : node.attr.sched.axis()) {
    if (i != *cur && cur != original.begin() && cur != original.end()) {
      throw std::runtime_error("Miss some axis in merge");
    }

    if (cur != original.end() && i == *cur) {
      cur++;
      if (cur == original.end()) {
        new_axis.push_back(merged_axis_id);
      }
    } else {
      new_axis.push_back(i);
    }
  }

  node.attr.sched.axis = new_axis;

  for (int i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    vector<AxisId> new_axis;
    vector<SizeExpr> new_repeat;
    vector<SizeExpr> new_strides;

    auto axis = node.outputs[i].axis();
    auto repeat = node.outputs[i].repeats();
    auto strides = node.outputs[i].strides();

    SizeExpr merge_repeat;

    auto cur = original.begin();
    for (int a = 0; a < axis.size(); a++) {
      if (axis[a] != *cur && cur != original.begin() && cur != original.end()) {
        throw std::runtime_error("Miss some axis in merge");
      }

      if (cur != original.end() && axis[a] == *cur) {
        cur++;
        merge_repeat *= repeat[a];
        if (cur == original.end()) {
          new_axis.push_back(merged_axis_id);
          new_repeat.push_back(merge_repeat);
          new_strides.push_back(strides[a]);
        }
      } else {
        new_axis.push_back(axis[a]);
        new_repeat.push_back(repeat[a]);
        new_strides.push_back(strides[a]);
      }
    }

    node.outputs[i].axis = new_axis;
    node.outputs[i].repeats = new_repeat;
    node.outputs[i].strides = new_strides;
  }
  return;
}

void Graph::ApplyMerge(NodeView &node, AxisId merged_axis_id) {
  if (merged_axis_id >= this->axis.Size()) {
    throw std::runtime_error("Merged axis id invalid.");
  }

  auto axis = this->axis[merged_axis_id];
  if (axis.type != ascir::Axis::AXIS_TYPE_MERGED) {
    throw std::runtime_error("Merged axis type invalid.");
  }

  ApplyMerge(node, merged_axis_id, axis.from);
}

void Graph::ApplyReorder(NodeView &node, const std::vector<AxisId> &reordered_axis) {
  auto node_axis = node.attr.sched.axis();
  if (node_axis.size() != reordered_axis.size()) {
    throw std::runtime_error("Miss some axis in reorder");
  }

  for (auto a : reordered_axis) {
    auto it = std::find(node_axis.begin(), node_axis.end(), a);
    if (it == node_axis.end()) {
      throw std::runtime_error("Miss reorder axis " + std::to_string(a) + "in node axis.");
    }
  }

  node.attr.sched.axis = reordered_axis;
  for (auto output : node.outputs()) {
    std::vector<AxisId> new_axis;
    std::vector<SizeExpr> new_repeat;
    std::vector<SizeExpr> new_strides;

    auto output_axis = output.axis();
    for (auto axis : reordered_axis) {
      auto it = std::find(output_axis.begin(), output_axis.end(), axis);
      if (it == output_axis.end()) {
        continue;
      }

      auto pos = std::distance(output_axis.begin(), it);
      new_axis.push_back(output_axis[pos]);
      new_repeat.push_back(output.repeats()[pos]);
      new_strides.push_back(output.strides()[pos]);
    }

    output.axis = new_axis;
    output.repeats = new_repeat;
    output.strides = new_strides;
  }
}

NodeView Graph::FindImpl(const char *name) const {
  auto graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  auto node = graph->FindNode(name);
  if (node == nullptr) {
    return NodeView{nullptr};
  }

  return NodeView{node};
}

NodeViewVisitor Graph::GetAllNodesImpl() const {
  auto graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (graph == nullptr) {
      return NodeViewVisitor{};
  }
  return NodeViewVisitor{graph->GetAllNodes()};
}

NodeViewVisitor Graph::GraphInputsImpl() const {
  auto graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (graph == nullptr) {
      return NodeViewVisitor{};
  }

  return NodeViewVisitor{graph->GetInputNodes()};
}

NodeViewVisitor Graph::GraphOutputsImpl() const {
  auto graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (graph == nullptr) {
      return NodeViewVisitor{};
  }

  return NodeViewVisitor{graph->GetOutputNodes()};
}

NodeView Graph::Find(const char *name) {
  return FindImpl(name);
}

NodeViewVisitor Graph::GetAllNodes() {
  return GetAllNodesImpl();
}

NodeViewVisitor Graph::GraphInputs() {
  return GraphInputsImpl();
}

NodeViewVisitor Graph::GraphOutputs() {
  return GraphOutputsImpl();
}

const NodeView Graph::Find(const char *name) const {
  return FindImpl(name);
}

NodeViewVisitorConst Graph::GetAllNodes() const {
  auto tmp = GetAllNodesImpl();
  return NodeViewVisitorConst(tmp);
}

NodeViewVisitorConst Graph::GraphInputs() const {
  auto tmp = GraphInputsImpl();
  return NodeViewVisitorConst(tmp);
}

NodeViewVisitorConst Graph::GraphOutputs() const {
  auto tmp = GraphOutputsImpl();
  return NodeViewVisitorConst(tmp);
}

int Graph::CopyFrom(const ascir::Graph &graph) {
  int ret = ge::GraphUtilsEx::CopyGraph(graph, *this);
  if (ret != 0) {
      return ret;
  }

  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (compute_graph == nullptr) {
    return 1;
  }
  this->holder = compute_graph.get();
  return 0;
}

void Graph::SortByExecOrder() {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  compute_graph->TopologicalSorting([](const ge::NodePtr &a, const ge::NodePtr &b) {
    const NodeView node_a(a);
    const NodeView node_b(b);
    return node_a.attr.sched.exec_order < node_b.attr.sched.exec_order;
  });
}
void Graph::AddNode(ge::Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto node = ge::GraphUtilsEx::GetComputeGraph(*this)->AddNode(op_desc);
  auto new_op = ge::OpDescUtils::CreateOperatorFromNode(node);
  std::swap(new_op, op);
}

void Graph::UpdateAxisAlign(AxisId id, int32_t align) {
  std::vector<int32_t> tmp;
  ge::AttrUtils::GetListInt(holder, this->axis.ALIGN, tmp);
  tmp[id] = align;
  ge::AttrUtils::SetListInt(holder, this->axis.ALIGN, tmp);
}

void Graph::UpdateAxisAllowOversizeAxis(AxisId id, bool allow_oversize_axis) {
  std::vector<bool> tmp;
  ge::AttrUtils::GetListBool(holder, this->axis.ALLOW_OVERSIZE_AXIS, tmp);
  tmp[id] = allow_oversize_axis;
  ge::AttrUtils::SetListBool(holder, this->axis.ALLOW_OVERSIZE_AXIS, tmp);
}

void Graph::UpdateAxisAllowUnalignedTail(AxisId id, bool allow_unaligned_tail) {
  std::vector<bool> tmp;
  ge::AttrUtils::GetListBool(holder, this->axis.ALLOW_UNALIGNED_TAIL, tmp);
  tmp[id] = allow_unaligned_tail;
  ge::AttrUtils::SetListBool(holder, this->axis.ALLOW_UNALIGNED_TAIL, tmp);
}

NodeViewIter::NodeViewIter(NodeViewVisitor::Iterator &&iter) : ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator(iter) {}

NodeViewIter &NodeViewIter::operator++() {
  ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator::operator++();
  return *this;
}

NodeView NodeViewIter::operator*() {
  auto ptr = ge::ComputeGraph::Vistor<ge::NodePtr>::Iterator::operator*();
  return NodeView{ptr};
}

bool NodeViewIter::operator!=(const NodeViewIter &other) const {
  return static_cast<const NodeViewVisitor::Iterator &>(*this) != static_cast<const NodeViewVisitor::Iterator &>(other);
}

NodeViewVisitor::NodeViewVisitor()
    : ge::ComputeGraph::Vistor<ge::NodePtr>(nullptr, vector<ge::NodePtr>{})
{}

NodeViewVisitor::NodeViewVisitor(ge::ComputeGraph::Vistor<ge::NodePtr> &&visitor)
    : ge::ComputeGraph::Vistor<ge::NodePtr>(visitor) {}

NodeViewIter NodeViewVisitor::begin() {
  return NodeViewIter(ge::ComputeGraph::Vistor<ge::NodePtr>::begin());
}

NodeViewIter NodeViewVisitor::end() {
  return NodeViewIter(ge::ComputeGraph::Vistor<ge::NodePtr>::end());
}

NodeViewIterConst &NodeViewIterConst::operator++() {
  NodeViewIter::operator++();
  return *this;
}

const NodeView NodeViewIterConst::operator*() {
  return NodeViewIter::operator*();
}

bool NodeViewIterConst::operator!=(const NodeViewIterConst &other) const {
  return NodeViewIter::operator!=(other);
}

NodeViewIterConst NodeViewVisitorConst::begin() {
  auto tmp = NodeViewVisitor::begin();
  return NodeViewIterConst(tmp);
}

NodeViewIterConst NodeViewVisitorConst::end() {
  auto tmp = NodeViewVisitor::end();
  return NodeViewIterConst(tmp);
}

namespace ascir {
void AddEdgeForNode(const ge::Operator &src_op, int32_t src_index, const ge::Operator &dst_op, int32_t dst_index) {
  auto src_node = ge::NodeUtilsEx::GetNodeFromOperator(src_op);
  auto dst_node = ge::NodeUtilsEx::GetNodeFromOperator(dst_op);
  if (src_node == nullptr || dst_node == nullptr) {
    return;
  }
  auto ret = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index));
  if (ret != ge::GRAPH_SUCCESS) {
    throw std::invalid_argument("Invalid src or dst index");
  }
}
}
