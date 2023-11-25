#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_

#include <map>
#include <mutex>

#include "external/graph/graph.h"
#include "ge/ge_graph_compile_summary.h"
#include "tng_status.h"
#include "ge/ge_allocator.h"

namespace tng {
class Session {
 public:
  static Session &GetInstance() {
    static Session instance;
    return instance;
  }

  Status Initialize(const std::map<std::string, std::string> &options);

  Status EnsureInitialized();

  Status Finalize();

  Status AddGraph(uint32_t id, const ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options);

  Status CompileGraph(uint32_t id, std::shared_ptr<ge::CompiledGraphSummary> *summary);

  Status AutoTuneGraph(const ge::Graph &graph, const std::map<ge::AscendString, ge::AscendString> &options,
                       const std::vector<ge::Tensor> &example_inputs, void *stream = nullptr);

  Status RemoveGraph(uint32_t id);

  Status RegisterExternalAllocator(const void *const stream, std::shared_ptr<ge::Allocator> allocator);

  Status RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                  void *stream = nullptr);

  Status SetGraphConstMemoryBase(uint32_t id, const void *const memory, size_t size);

  Status UpdateGraphFeatureMemoryBase(uint32_t id, const void *const memory, size_t size);

 private:
  Session() : initialized_(false), status_(Status::Success()){};
  std::mutex mu_;
  std::atomic_bool initialized_;
  std::atomic_bool run_with_torch_npu_ = false;
  Status status_;
  int32_t device_index_ = -1;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_