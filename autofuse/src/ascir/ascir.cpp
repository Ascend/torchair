#include "ascir.h"

#include "graph_utils_ex.h"

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
  holder = &tmp_attr_holder;
}

Graph &Graph::SetInputs(const std::vector<ge::Operator> &inputs) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (compute_graph != nullptr) {
    ge::AttrUtils::ClearAllAttrs(this->tmp_attr_holder);
    this->tmp_attr_holder.CopyAttrsFrom(*compute_graph);
    this->holder = &this->tmp_attr_holder;
  }

  // Will build new compute_graph after set inputs
  ge::Graph::SetInputs(inputs);
  compute_graph = ge::GraphUtilsEx::GetComputeGraph(*this);
  if (compute_graph == nullptr) {
    return *this;
  }

  compute_graph->CopyAttrsFrom(*this->holder);

  this->holder = compute_graph.get();
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

  vector<string> axis_names;
  vector<Axis::Type> axis_types;
  vector<vector<SizeVarId>> axis_size_num;
  vector<vector<SizeVarId>> axis_size_den;
  vector<vector<AxisId>> axis_from;

  ge::AttrUtils::GetListStr(this->holder, this->axis.NAME, axis_names);
  ge::AttrUtils::GetListInt(this->holder, this->axis.TYPE, axis_types);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.SIZE_NUMS, axis_size_num);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.SIZE_DENS, axis_size_den);
  ge::AttrUtils::GetListListInt(this->holder, this->axis.FROM, axis_from);

  axis_names.push_back(name);
  axis_types.push_back(type);
  axis_size_num.push_back(size.nums);
  axis_size_den.push_back(size.dens);
  axis_from.push_back(from);

  ge::AttrUtils::SetListStr(this->holder, this->axis.NAME, axis_names);
  ge::AttrUtils::SetListInt(this->holder, this->axis.TYPE, axis_types);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.SIZE_NUMS, axis_size_num);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.SIZE_DENS, axis_size_den);
  ge::AttrUtils::SetListListInt(this->holder, this->axis.FROM, axis_from);

  ge::AttrUtils::SetInt(this->holder, this->axis.NUM, num_of_axis + 1);
  return new_axis;
}

Axis Graph::CreateAxis(const std::string &name, const SizeExpr &size) {
  return CreateAxis(name, Axis::AXIS_TYPE_ORIGINAL, size, {});
}

std::tuple<Axis, Axis> Graph::BlockSplit(AxisId axis_id) {
  auto axis = this->axis[axis_id];
  auto block_size = this->CreateSizeVar(axis.name + "b_size", SizeVar::SIZE_TYPE_VAR, 0);

  auto outter = this->CreateAxis(axis.name + "B", Axis::AXIS_TYPE_BLOCK_OUTER,
                                 axis.size / SizeExpr{{block_size.id}}, {axis_id});
  auto inner = this->CreateAxis(axis.name + "b",
                                Axis::AXIS_TYPE_BLOCK_INNER,
                                SizeExpr{{block_size.id}}, {axis_id});
  return std::make_tuple(outter, inner);
}

std::tuple<Axis, Axis> Graph::TileSplit(AxisId axis_id) {
  auto axis = this->axis[axis_id];
  auto tile_size = this->CreateSizeVar(axis.name + "t_size", SizeVar::SIZE_TYPE_VAR, 0);

  auto outter = this->CreateAxis(axis.name + "T",
                                 Axis::AXIS_TYPE_TILE_OUTER,
                                 axis.size / SizeExpr{{tile_size.id}}, {axis_id});
  auto inner = this->CreateAxis(axis.name + "t",
                                Axis::AXIS_TYPE_TILE_INNER,
                                SizeExpr{{tile_size.id}}, {axis_id});
  return std::make_tuple(outter, inner);
}

Axis Graph::MergeAxis(std::initializer_list<AxisId> axis) {
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

void Graph::ApplySplit(NodeView &node, AxisId outter, AxisId inner, AxisId original) {
  SizeExpr split_size{this->axis[inner].size};

  vector<AxisId> new_axis;
  auto axis = node.attr.sched.axis();
  for (auto &i : axis) {
    if (i == original) {
      new_axis.push_back(outter);
      new_axis.push_back(inner);
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
        new_axis.push_back(outter);
        new_repeat.push_back(repeat[a] / split_size);
        new_strides.push_back(strides[a] * split_size);

        new_axis.push_back(inner);
        new_repeat.push_back(split_size);
        new_strides.push_back(strides[a]);
      }
    }

    node.outputs[i].axis = new_axis;
    node.outputs[i].repeats = new_repeat;
    node.outputs[i].strides = new_strides;
  }

  return;
}

void Graph::ApplyMerge(NodeView &node, AxisId merged_axis, const std::vector<AxisId> &original) {
  vector<AxisId> new_axis;

  auto cur = original.begin();
  for (auto i : node.attr.sched.axis()) {
    if (i != *cur && cur != original.begin() && cur != original.end()) {
      throw std::runtime_error("Miss some axis in merge");
    }

    if (i == *cur) {
      cur++;
    } else {
      new_axis.push_back(i);
    }

    if (cur == original.end()) {
      new_axis.push_back(merged_axis);
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

      if (axis[a] == *cur) {
        cur++;
        merge_repeat *= repeat[a];
      } else {
        new_axis.push_back(a);
      }

      if (cur == original.end()) {
        new_axis.push_back(merged_axis);
        new_repeat.push_back(merge_repeat);
        new_strides.push_back(strides[a]);
      }
    }

    node.outputs[i].axis = new_axis;
    node.outputs[i].repeats = new_repeat;
    node.outputs[i].strides = new_strides;
  }
  return;
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
      this->tmp_attr_holder.CopyAttrsFrom(graph.tmp_attr_holder);
      this->holder = &this->tmp_attr_holder;
  } else {
      ge::AttrUtils::ClearAllAttrs(this->tmp_attr_holder);
      this->holder = compute_graph.get();
  }
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
