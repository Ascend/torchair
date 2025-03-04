#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXECUTOR_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXECUTOR_H_

#include <memory>
#include <vector>

#include "logger.h"
#include "tng_status.h"
#include "torch/torch.h"

namespace tng {
enum ExecutorStage : int32_t {
  kBegin,
  kPre,
  kAssembleInputs,
  kAssembleOutputs,
  kRunGraph,
  kStageCount
};

class GraphData;
class Executor {
 public:
  static Status Create(const std::shared_ptr<GraphData> &graph_data, std::unique_ptr<Executor> &executor);
  virtual ~Executor() = default;
  virtual Status Run(const std::vector<at::Tensor> &torch_inputs,
                     const std::vector<c10::optional<at::Tensor>> &torch_outputs, std::vector<at::Tensor> &outputs,
                     void *stream) = 0;

  using Creator = std::function<Status(const std::shared_ptr<GraphData> &, std::unique_ptr<Executor> &)>;
  static bool RegisterExecutorCreator(const Creator &creator, int32_t priority = 0);

 protected:
  Executor() = default;
  std::map<ExecutorStage, uint64_t> stages;

  void SetStageTime(ExecutorStage stage) {
    auto time = tng::GetTimestampForEventLog();
    if (time != 0) {
      stages[stage] = time;
    }
  }

  std::string GenEventLog() {
    if (stages.size() != ExecutorStage::kStageCount) {
      return "log error";
    }
    std::ostringstream oss;
    oss << "ge run graph at " << stages[ExecutorStage::kAssembleOutputs]
        << ", pre process: " << stages[ExecutorStage::kPre] - stages[ExecutorStage::kBegin]
        << "us, assemble input: " << stages[ExecutorStage::kAssembleInputs] - stages[ExecutorStage::kPre]
        << "us, assemble output: " << stages[ExecutorStage::kAssembleOutputs] - stages[ExecutorStage::kAssembleInputs]
        << "us, run graph: " << stages[ExecutorStage::kRunGraph] - stages[ExecutorStage::kAssembleOutputs] << "us";

    stages.clear();
    return oss.str();
  }

 private:
  static std::map<int, Executor::Creator> creators_;
  static std::mutex mutex_;
};

#define REGISTER_EXECUTOR_CREATOR(creator, priority)                          \
  static bool __attribute__((unused)) __register_executor_creator_##creator = \
    tng::Executor::RegisterExecutorCreator(creator, priority)
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_EXECUTOR_H_