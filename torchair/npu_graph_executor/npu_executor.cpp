#include "external/graph/types.h"
#include "executor.h"
#include "static_npu_graph_executor.h"
#include "dynamic_npu_graph_executor.h"
#include "muti_gear_npu_graph_executor.h"

#include <utility>
#include "graph_data.h"
#include "graph/tensor.h"
#include "torch/torch.h"
#include "session.h"
#include "checker.h"
#include "utils.h"
#include "logger.h"

namespace {
bool IsGearGraph(const std::shared_ptr<tng::GraphData> &graph_data) {
  std::vector<const char*> gears_options = {
    "ge.dynamicDims",
    "ge.inputShape",
    "ge.dynamicNodeType"
  };
  for (const char* option : gears_options) {
    auto tmp = graph_data->compile_options.find(ge::AscendString(option));
    if (tmp == graph_data->compile_options.end()) {
      return false;
    }
  }
  return true;
}
}

namespace tng {
namespace {
Status CreateNpuGraphExecutor(const std::shared_ptr<GraphData> &graph_data, std::unique_ptr<Executor> &executor) {
  TNG_ASSERT_NOTNULL(graph_data->summary);
  if (graph_data->summary->IsStatic()) {
    if (IsGearGraph(graph_data)) {
      TNG_LOG(INFO) << "Create muti gear npu graph executor for graph " << graph_data->id;
      executor = std::make_unique<MutiGearNpuGraphExecutor>(graph_data);
    } else {
      TNG_LOG(INFO) << "Create static npu graph executor for graph " << graph_data->id;
      executor = std::make_unique<StaticNpuGraphExecutor>(graph_data);
    }
  } else {
    TNG_LOG(INFO) << "Create dynamic npu graph executor for graph " << graph_data->id;
    executor = std::make_unique<DynamicNpuGraphExecutor>(graph_data);
  }
  return Status::Success();
}
}  // namespace

REGISTER_EXECUTOR_CREATOR(CreateNpuGraphExecutor, 0);
}  // namespace tng