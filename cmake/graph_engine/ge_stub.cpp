#include <iostream>

#include "external/graph/types.h"
#include "ge/ge_api.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"

namespace ge {
class CompiledGraphSummary::SummaryData {
 public:
  SummaryData() = default;
  ~SummaryData() = default;
  bool IsStatic() const { return is_static_; }
  bool IsFeatureMemoryBaseRefreshable() const { return is_feature_mem_refreshable_; }
  size_t GetConstMemorySize() const { return const_mem_size_; }
  size_t GetFeatureMemorySize() const { return feature_mem_size_; }
  size_t GetStreamNum() const { return stream_num_; }
  size_t GetEventNum() const { return event_num_; }
  std::vector<ge::Shape> GetOutputShapes() { return netoutput_shapes_; }

 private:
  bool is_static_{false};
  bool is_feature_mem_refreshable_{false};
  size_t const_mem_size_{512UL};
  size_t feature_mem_size_{10240UL};
  size_t stream_num_{1UL};
  size_t event_num_{2UL};
  std::vector<ge::Shape> netoutput_shapes_;
};

class CompiledGraphSummary::Builder {
 public:
  Builder() = default;
  ~Builder() = default;
  static CompiledGraphSummaryPtr Build() {
    CompiledGraphSummaryPtr summary(new CompiledGraphSummary);
    summary->data_ = std::make_shared<SummaryData>();
    return summary;
  }
};

CompiledGraphSummary::~CompiledGraphSummary() = default;
bool CompiledGraphSummary::IsStatic() const { return data_->IsStatic(); }
Status CompiledGraphSummary::GetFeatureMemoryBaseRefreshable(bool &v) const {
  v = data_->IsFeatureMemoryBaseRefreshable();
  return SUCCESS;
}
Status CompiledGraphSummary::GetConstMemorySize(size_t &size) const {
  size = data_->GetConstMemorySize();
  return SUCCESS;
}
Status CompiledGraphSummary::GetFeatureMemorySize(size_t &size) const {
  size = data_->GetFeatureMemorySize();
  return SUCCESS;
}
Status CompiledGraphSummary::GetStreamNum(size_t &num) const {
  num = data_->GetStreamNum();
  return SUCCESS;
}
Status CompiledGraphSummary::GetEventNum(size_t &num) const {
  num = data_->GetEventNum();
  return SUCCESS;
}

std::string GEGetErrorMsg() { return "[STUB] Something error"; }

Session::Session(const std::map<AscendString, AscendString> &options) {
  std::cerr << "[STUB] Session::Session created" << std::endl;
}

Session::~Session() { std::cerr << "[STUB] Session::Session destroyed" << std::endl; }

namespace {
std::map<uint32_t, size_t> graph_id_to_num_outputs;
}

Status Session::AddGraph(uint32_t id, const ge::Graph &graph, const std::map<AscendString, AscendString> &options) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    graph_id_to_num_outputs[id] = 0U;
  } else {
    auto netoutput = compute_graph->FindFirstNodeMatchType("NetOutput");
    if (netoutput == nullptr) {
      graph_id_to_num_outputs[id] = 0U;
    } else {
      graph_id_to_num_outputs[id] = netoutput->GetAllInDataAnchorsSize();
    }
  }
  std::cerr << "[STUB] Session::AddGraph graph " << id << ", num outputs " << graph_id_to_num_outputs[id] << std::endl;
  return ge::SUCCESS;
}

Status Session::CompileGraph(uint32_t id) {
  std::cerr << "[STUB] Session::CompileGraph graph " << id << std::endl;
  return ge::SUCCESS;
}

std::shared_ptr<ge::CompiledGraphSummary> Session::GetCompiledGraphSummary(uint32_t id) {
  std::cerr << "[STUB] Session::GetCompiledGraphSummary graph " << id << std::endl;
  return CompiledGraphSummary::Builder::Build();
}

Status Session::RemoveGraph(uint32_t id) {
  std::cerr << "[STUB] Session::RemoveGraph graph " << id << std::endl;
  return ge::SUCCESS;
}

Status Session::RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  std::cerr << "[STUB] Session::RunGraph graph " << id << std::endl;
  for (size_t i = 0; i < graph_id_to_num_outputs[id]; ++i) {
    ge::Tensor output;
    ge::TensorDesc desc;
    desc.SetDataType(ge::DT_FLOAT);
    desc.SetShape(ge::Shape(std::vector<int64_t>{512, 1024, 1024}));
    output.SetTensorDesc(desc);

    static std::vector<float> data;
    data.resize(512 * 1024 * 1024, 1.0);
    output.SetData(reinterpret_cast<uint8_t *>(data.data()), sizeof(float) * data.size());
    outputs.push_back(output);
  }
  return ge::SUCCESS;
}

Status Session::RunGraphWithStreamAsync(uint32_t id, void *stream, const std::vector<ge::Tensor> &inputs,
                                        std::vector<ge::Tensor> &outputs) {
  std::cerr << "[STUB] Session::RunGraphWithStreamAsync graph " << id << std::endl;
  return ge::SUCCESS;
}

Status Session::RegisterExternalAllocator(const void *const stream, std::shared_ptr<ge::Allocator> allocator) const {
  (void)stream;
  (void)allocator;
  return ge::SUCCESS;
}

Status Session::UnregisterExternalAllocator(const void *const stream) const {
  (void)stream;
  return ge::SUCCESS;
}

Status GEInitialize(const std::map<ge::AscendString, ge::AscendString> &options) {
  std::cerr << "[STUB] GEInitialize" << std::endl;
  return ge::SUCCESS;
}
Status GEFinalize() {
  std::cerr << "[STUB] GEFinalize" << std::endl;
  return ge::SUCCESS;
}
}  // namespace ge