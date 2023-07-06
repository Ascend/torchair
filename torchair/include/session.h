#ifndef TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_
#define TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_

#include <map>
#include <mutex>

#include "external/graph/graph.h"
#include "ge/ge_graph_compile_summary.h"
#include "tng_status.h"

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

  Status RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs,
                  void *stream = nullptr);

 private:
  Session() : initialized_(false), status_(Status::Success()){};
  std::mutex mu_;
  std::atomic_bool initialized_;
  Status status_;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CONCRETE_GRAPH_SESSION_H_