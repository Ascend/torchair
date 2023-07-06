#include "external/graph/types.h"
#include "executor.h"
#include "npu_graph_executor.h"

#include <utility>
#include "graph_data.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "torch/torch.h"
#include "session.h"
#include "checker.h"
#include "utils.h"
#include "logger.h"

namespace tng {
namespace {
Status CreateNpuGraphExecutor(const std::shared_ptr<GraphData> &graph_data, std::unique_ptr<Executor> &executor) {
  TNG_ASSERT_NOTNULL(graph_data->summary);
  if (graph_data->summary->IsStatic()) {
    TNG_LOG(INFO) << "Create static npu graph executor for graph " << graph_data->id;
    executor = std::unique_ptr<Executor>(new StaticNpuGraphExecutor(graph_data));
  } else {
    TNG_LOG(INFO) << "Create dynamic graph executor for graph " << graph_data->id;
    // TODO: Create dynmaic graph executor for npu here
  }
  return Status::Success();
}
}  // namespace

REGISTER_EXECUTOR_CREATOR(CreateNpuGraphExecutor, 0);
}  // namespace tng