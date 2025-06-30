#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_

#include <map>
#include <unistd.h>

#include "executor.h"
#include "export.h"
#include "external/graph/ascend_string.h"
#include "external/graph/types.h"
#include "graph/tensor.h"
#include "session.h"
#include "tng_status.h"
#include "torch/torch.h"
#include "logger.h"
#include "acl/acl_rt.h"

namespace tng {
class GraphData;
class NpuConcreteGraph {
 public:
  static Status Create(const void *serialized_proto, size_t proto_size,
                       const std::map<ge::AscendString, ge::AscendString> &options,
                       std::vector<int64_t> input_placements, std::vector<int64_t> output_dtypes, int64_t executor_type,
                       std::unique_ptr<NpuConcreteGraph> &graph);

  ~NpuConcreteGraph() {
    if (created_pid_ != getpid()) {
      TNG_LOG(DEBUG) << "Skip release graph " << graph_id_ << ", because graph is created on pid: " << created_pid_
                     << ", but try to release on pid: " << getpid();
      return;
    }

    // When ge session is not initialized or graph is unloaded, remove graph is forbidden.
    if (!is_graph_added_) {
      TNG_LOG(DEBUG) << "Skip release graph " << graph_id_ << ", because graph have not been added.";
      return;
    }

    if (!Session::GetInstance().IsInitialized()) {
      TNG_LOG(DEBUG) << "Skip release graph " << graph_id_ << ", because session is not initialized.";
      return;
    }

    if (executor_ != nullptr) {
      auto bind_stream = executor_->GetBindStream();
      if (bind_stream != nullptr) {
        const auto ret = aclrtSynchronizeStream(bind_stream);
        TNG_LOG(DEBUG) << "Before release graph" << graph_id_ << ", synchronize stream return value is " << ret;
      }
    }

    (void)Session::GetInstance().RemoveGraph(graph_id_);
    is_graph_added_ = false;
  }

  Status Compile();
  Status AutoTune(const std::vector<at::Tensor> &example_inputs, void *stream = nullptr);
  Status SetHintShape(const std::vector<std::vector<int64_t>> &inputs_shape,
                      const std::vector<std::vector<int64_t>> &outputs_shape);
  Status Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs, std::vector<at::Tensor> &outputs,
             void *stream = nullptr);

  Status AssembleInputs(const std::vector<const at::Tensor *> &tensors);

  static Status ReleaseResource();
  static Status InitializeResource(const std::map<std::string, std::string> &options);

 private:
  NpuConcreteGraph(std::shared_ptr<GraphData> graph_data);
  std::shared_ptr<GraphData> graph_data_ = nullptr;
  std::unique_ptr<Executor> executor_ = nullptr;

  bool is_graph_added_{false};
  uint32_t graph_id_{0};
  pid_t created_pid_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_CONCRETE_GRAPH_H_
