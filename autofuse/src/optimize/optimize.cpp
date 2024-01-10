#include "optimize.h"

#include "ascir_ops.h"
#include "autoschedule/autoschedule.h"

using namespace ascir;
using namespace optimize;

Optimizer::Optimizer(const OptimizerOptions &options) {}

int Optimizer::Optimize(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  // 对dtype和stride的推导要放在原图上
  // 这样原图自身才是dtype和stirde连续的
  // 这一步本身不是优化而是图的完整性准备。
  int ret = this->InferOutput(graph);
  if (ret != 0) {
    return ret;
  }

  ascir::ImplGraph optimize_graph(graph.GetName().c_str());
  optimize_graph.CopyFrom(graph);

  ret = this->GetApiInfo(graph, optimize_graph);
  if (ret != 0) {
    return ret;
  }

  ret = this->AutoScheduler(graph, optimize_graph, optimize_graphs);
  if (ret != 0) {
    return ret;
  }

  ret = this->SelectLoopAxis(graph,  optimize_graphs);
  if (ret != 0) {
    return ret;
  }

  ret = this->BufQueAlloc(graph, optimize_graphs);
  if (ret != 0) {
    return ret;
  }
  return 0;
}

static void DataInferOutput(ascir::NodeView &node) {
  node.attr.hint.compute_type = COMPUTE_DATA;
}

static void ElemwiseUniaryInferOutput(ascir::NodeView &node) {
  node.attr.hint.compute_type = COMPUTE_ELEWISE;
  node.outputs[0].dtype = (ge::DataType)node.inputs[0]->dtype;
  node.outputs[0].axis = node.inputs[0]->axis();
  node.outputs[0].repeats = node.inputs[0]->repeats();
  node.outputs[0].strides = node.inputs[0]->strides();
}

static void LoadInferOutput(ascir::NodeView &node) {
  node.attr.hint.compute_type = COMPUTE_LOAD;
  node.outputs[0].dtype = (ge::DataType)node.inputs[0]->dtype;
}

static void StoreInferOutput(ascir::NodeView &node) {
  node.attr.hint.compute_type = COMPUTE_STORE;
  node.outputs[0].dtype = (ge::DataType)node.inputs[0]->dtype;
}

int Optimizer::InferOutput(ascir::HintGraph &graph) {
  const map<std::string, std::function<void(NodeView&)>> infer_funcs = {
    {ops::Data::Type, &DataInferOutput},
    {ops::Output::Type, &DataInferOutput},
    {ops::Load::Type, &LoadInferOutput},
    {ops::Store::Type, &StoreInferOutput},
    {ops::Abs::Type, &ElemwiseUniaryInferOutput}
  };

  for (auto node : graph.GetAllNodes()) {
    auto it = infer_funcs.find(node->GetType().c_str());
    if (it != infer_funcs.end()) {
      it->second(node);
    }
  }

  return 0;
}

static void DataGetApiInfo(ascir::NodeView &node) {
  node.attr.api.type = API_TYPE_BUFFER;
  node.attr.api.unit = UNIT_NONE;
}

static void VectorGetApiInfo(ascir::NodeView &node) {
  node.attr.api.type = API_TYPE_COMPUTE;
  node.attr.api.unit = UNIT_VECTOR;
}

static void MteGetApiInfo(ascir::NodeView &node) {
  node.attr.api.type = API_TYPE_COMPUTE;
  node.attr.api.unit = UNIT_MTE;
}

int Optimizer::GetApiInfo(const ascir::HintGraph &graph, ascir::ImplGraph &optimize_graph) {
  const map<std::string, std::function<void(NodeView &)>> get_funcs = {
    {ops::Data::Type, &DataGetApiInfo},
    {ops::Output::Type, &DataGetApiInfo},
    {ops::Load::Type, &MteGetApiInfo},
    {ops::Store::Type, &MteGetApiInfo},
    {ops::Abs::Type, &VectorGetApiInfo}
  };

  for (auto node : optimize_graph.GetAllNodes()) {
    auto it = get_funcs.find(node->GetType().c_str());
    if (it!= get_funcs.end()) {
      it->second(node);
    }
  }
  return 0;
}

int Optimizer::AutoScheduler(const ascir::HintGraph& graph, const ascir::ImplGraph& optimizer_graph, std::vector<ascir::ImplGraph> &impl_graphs) {
  auto scheduler = autoschedule::AutoSchedule(optimizer_graph, impl_graphs);
  scheduler.DoAutoSchedule();
  return 0;
}

int Optimizer::SelectLoopAxis(const ascir::HintGraph& graph, std::vector<ascir::ImplGraph> &impl_graphs) {
  for (auto impl_graph : impl_graphs) {
    for (auto node : impl_graph.GetAllNodes()) {
      node.attr.sched.loop_axis = ID_NONE;
      if (node.attr.hint.compute_type == COMPUTE_DATA) {
        continue;
      }

      auto axis = node.attr.sched.axis();
      for (auto output : node.outputs()) {
        for (auto vectorize_axis : output.vectorized_axis()) {
          auto it = std::find(axis.begin(), axis.end(), vectorize_axis);
          if (it != axis.end()) {
            *it = ID_NONE;
          }
        }
      }

      for (auto a = axis.rbegin(); a != axis.rend(); ++a) {
        if (*a != ID_NONE) {
          node.attr.sched.loop_axis = *a;
          break;
        }
      }

      if (node.attr.sched.loop_axis == ID_NONE) {
        throw std::runtime_error("Can not find loop axis");
      }
    }
  }

  return 0;
}

int Optimizer::BufQueAlloc(const ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &optimize_graphs) {
  int tensor_id = 0;
  int que_id = 0;
  int buf_id = 0;

  for (auto impl_grpah : optimize_graphs) {
    for (auto node : impl_grpah.GetAllNodes()) {
      if (ops::IsOps<ops::Data>(node) || ops::IsOps<ops::Output>(node) || ops::IsOps<ops::Store>(node)) {
        node.outputs[0].mem.tensor_id = tensor_id++;
        node.outputs[0].mem.alloc_type = ALLOC_TYPE_GLOBAL;
        node.outputs[0].mem.hardware = MEM_HARDWARE_UB;
        node.outputs[0].mem.position = POSITION_GM;
        node.outputs[0].buf.id = ID_NONE;
        node.outputs[0].que.id = ID_NONE;
        continue;
      }

      for (auto output : node.outputs()) {
        output.mem.tensor_id = tensor_id++;
        if (ops::IsOps<ops::Load>(node)) {
          output.mem.position = POSITION_VECIN;
        } else {
          output.mem.position = POSITION_VECOUT;
        }

        // Using que/buf according to if tensor is used by other unit
        bool output_use_by_other_unit = false;
        for (auto input : output->GetPeerInDataAnchors()) {
          auto peer = NodeView(input->GetOwnerNode());
          if (node.attr.api.unit != peer.attr.api.unit) {
            output_use_by_other_unit = true;
            break;
          }
        }

        if (output_use_by_other_unit) {
          output.mem.alloc_type = ALLOC_TYPE_QUEUE;
          output.buf.id = ID_NONE;
          output.que.id = que_id++;
          output.que.depth = 2;
          output.que.buf_num = 2;
        } else {
          output.mem.alloc_type = ALLOC_TYPE_BUFFER;
          output.buf.id = buf_id++;
          output.que.id = ID_NONE;
        }

        // Currently, we don't support inplace/merge optimization
        output.opt.ref_tensor = ID_NONE;
        output.opt.merge_scope = ID_NONE;
      }
    }
  }

  return 0;
}
