#include <cstdint>
#include <iostream>
#include <mutex>

#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_api.h"
#include "graph/types.h"

namespace {
constexpr size_t kOutptSize = 512 * 1024 * 1024;
}

namespace ge {
class CompiledGraphSummary::SummaryData {
 public:
  SummaryData() = default;
  ~SummaryData() = default;
  bool IsStatic() const {
    return is_static_;
  }
  bool IsFeatureMemoryBaseRefreshable() const {
    return is_feature_mem_refreshable_;
  }
  size_t GetConstMemorySize() const {
    return const_mem_size_;
  }
  size_t GetFeatureMemorySize() const {
    return feature_mem_size_;
  }
  size_t GetFixedFeatureMemorySize() const {
    return fixed_mem_size_;
  }
  size_t GetRefreshableFeatureMemorySize() const {
    return refreshable_mem_size_;
  }
  size_t GetStreamNum() const {
    return stream_num_;
  }
  size_t GetEventNum() const {
    return event_num_;
  }
  std::vector<ge::Shape> GetOutputShapes() {
    return netoutput_shapes_;
  }
  std::vector<ge::DataType> GetOutputDtypes() {
    return output_dtypes_;
  }

  bool is_static_{false};
  bool is_feature_mem_refreshable_{false};
  size_t const_mem_size_{512UL};
  size_t feature_mem_size_{10240UL};
  size_t fixed_mem_size_{512UL};
  size_t refreshable_mem_size_{10240UL};
  size_t stream_num_{1UL};
  size_t event_num_{2UL};
  std::vector<ge::Shape> netoutput_shapes_;
  std::vector<ge::DataType> output_dtypes_;
};

class CompiledGraphSummary::Builder {
 public:
  Builder() = default;
  ~Builder() = default;
  static CompiledGraphSummaryPtr Build(const CompiledGraphSummary::SummaryData &data) {
    CompiledGraphSummaryPtr summary(new CompiledGraphSummary);
    summary->data_ = std::make_shared<SummaryData>();
    *summary->data_ = data;
    return summary;
  }
};

CompiledGraphSummary::~CompiledGraphSummary() = default;
bool CompiledGraphSummary::IsStatic() const {
  return data_->IsStatic();
}
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
Status CompiledGraphSummary::GetFixedFeatureMemorySize(size_t &size) const {
  size = data_->GetFixedFeatureMemorySize();
  return SUCCESS;
}

Status CompiledGraphSummary::GetRefreshableFeatureMemorySize(size_t &size) const {
  size = data_->GetRefreshableFeatureMemorySize();
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
Status CompiledGraphSummary::GetOutputShapes(std::vector<ge::Shape> &shapes) const {
  shapes = data_->GetOutputShapes();
  return SUCCESS;
}
Status CompiledGraphSummary::GetOutputDtypes(std::vector<ge::DataType> &dtypes) const {
  dtypes = data_->GetOutputDtypes();
  return SUCCESS;
}

std::string GEGetErrorMsg() {
  return "[STUB] Something error";
}

std::string GEGetWarningMsg() {
  return "[STUB] Something warn";
}

ge::AscendString GEGetWarningMsgV2() {
  std::string warning_msg = "[STUB] Something warn";
  return ge::AscendString(warning_msg.c_str(), warning_msg.length());
}

Session::Session(const std::map<AscendString, AscendString> &options) {
  std::cerr << "[STUB] Session::Session created" << std::endl;
}

Session::~Session() {
  std::cerr << "[STUB] Session::Session destroyed" << std::endl;
}

namespace {
class GraphSpecManager {
 public:
  static CompiledGraphSummary::SummaryData &GetSpec(uint32_t id) {
    return Instance().Get(id);
  }

  static CompiledGraphSummary::SummaryData &Add(uint32_t id, const ge::Graph &graph,
                                                const std::map<AscendString, AscendString> &options) {
    auto &spec = Instance().Get(id);

    spec.is_static_ = true;
    for (auto &node : graph.GetDirectNode()) {
      AscendString type;
      node.GetType(type);
      TensorDesc desc;
      if (type == "Data" || type == "RefData") {
        node.GetOutputDesc(0, desc);
        const auto &dims = desc.GetShape().GetDims();
        for (const auto i : dims) {
          if (i < 0) {
            spec.is_static_ = false;
            break;
          }
        }
      } else if (type == "NetOutput") {
        size_t input_size = node.GetInputsSize();
        for (size_t i = 0u; i < input_size; ++i) {
          node.GetInputDesc(i, desc);
          spec.output_dtypes_.push_back(desc.GetDataType());
          spec.netoutput_shapes_.emplace_back(desc.GetShape());
        }
      }
    }

    auto iter = options.find("ge.featureBaseRefreshable");
    if (iter != options.end()) {
      spec.is_feature_mem_refreshable_ = iter->second == "1";
      std::cerr << "[STUB] Session::Add graph, is_feature_mem_refreshable_ = " << spec.is_feature_mem_refreshable_ << std::endl;
    }

    return spec;
  }

 private:
  GraphSpecManager() = default;
  static GraphSpecManager &Instance() {
    static GraphSpecManager instance;
    return instance;
  }
  CompiledGraphSummary::SummaryData &Get(uint32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return specs_[id];
  }  

  std::mutex mutex_;
  std::map<uint32_t, CompiledGraphSummary::SummaryData> specs_;
};
}  // namespace

Status Session::AddGraph(uint32_t id, const ge::Graph &graph, const std::map<AscendString, AscendString> &options) {
  auto &spec = GraphSpecManager::Add(id, graph, options);
  std::cerr << "[STUB] Session::AddGraph graph " << id << ", num outputs " << spec.output_dtypes_.size() << std::endl;
  return ge::SUCCESS;
}

Status Session::CompileGraph(uint32_t id) {
  std::cerr << "[STUB] Session::CompileGraph graph " << id << std::endl;
  return ge::SUCCESS;
}

std::shared_ptr<ge::CompiledGraphSummary> Session::GetCompiledGraphSummary(uint32_t id) {
  std::cerr << "[STUB] Session::GetCompiledGraphSummary graph " << id << std::endl;
  auto summary = CompiledGraphSummary::Builder::Build(GraphSpecManager::GetSpec(id));
  return summary;
}

Status Session::RemoveGraph(uint32_t id) {
  std::cerr << "[STUB] Session::RemoveGraph graph " << id << std::endl;
  return ge::SUCCESS;
}

Status Session::RunGraph(uint32_t id, const std::vector<ge::Tensor> &inputs, std::vector<ge::Tensor> &outputs) {
  std::cerr << "[STUB] Session::RunGraph graph " << id << std::endl;
  if (inputs.size() < 1U) {
    std::cerr << "[STUB] Input size is empty" << id << std::endl;
    return ge::SUCCESS;
  }
  auto spec = GraphSpecManager::GetSpec(id);
  for (size_t i = 0; i < spec.output_dtypes_.size(); ++i) {
    ge::Tensor output;
    ge::TensorDesc desc;
    desc.SetDataType(spec.output_dtypes_[i]);
    desc.SetShape(ge::Shape({2, 2}));
    output.SetTensorDesc(desc);

    static std::vector<float> data;
    data.resize(1024, 1.0);
    output.SetData(reinterpret_cast<uint8_t *>(data.data()), sizeof(float) * data.size());
    outputs.push_back(output);
  }
  return ge::SUCCESS;
}

Status Session::RunGraphWithStreamAsync(uint32_t id, void *stream, const std::vector<ge::Tensor> &inputs,
                                        std::vector<ge::Tensor> &outputs) {
  std::cerr << "[STUB] Session::RunGraphWithStreamAsync graph " << id << std::endl;
  std::cerr << "[STUB] Input size is " << inputs.size() << std::endl;
  if (inputs.size() < 1U) {
    std::cerr << "[STUB] Input size is empty " << id << std::endl;
    return ge::SUCCESS;
  }

  Placement placement = ge::Placement::kPlacementHost;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].GetTensorDesc().GetPlacement() != ge::Placement::kPlacementHost) {
      placement = ge::Placement::kPlacementDevice;
      break;
    }
  }
  outputs.clear();
  auto spec = GraphSpecManager::GetSpec(id);
  for (size_t i = 0; i < spec.output_dtypes_.size(); ++i) {
    ge::Tensor output;
    ge::TensorDesc desc;
    desc.SetDataType(spec.output_dtypes_[i]);
    char *st_muti_gear_stub = std::getenv("ST_GEARS_STUB_OUTPUTSHAPE");
    if (st_muti_gear_stub != NULL) {
      std::vector<int64_t> dims;
      dims.resize(inputs[0].GetShapeDimNum());
      for (size_t i = 0U; i < dims.size(); ++i) {
        dims[i] = inputs[0].GetShapeDim(i);
      }
      desc.SetShape(ge::Shape(dims));
    } else {
      desc.SetShape(spec.netoutput_shapes_[i]);
    }
    desc.SetPlacement(placement);
    output.SetTensorDesc(desc);

    static std::vector<float> data;
    data.resize(kOutptSize, 0.0);
    output.SetData(reinterpret_cast<uint8_t *>(data.data()), sizeof(float) * data.size());
    outputs.push_back(output);
  }
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

Status Session::SetGraphConstMemoryBase(uint32_t id, const void *const memory, size_t size) {
  (void)id;
  (void)memory;
  (void)size;
  return ge::SUCCESS;
}

Status Session::UpdateGraphFeatureMemoryBase(uint32_t id, const void *const memory, size_t size) {
  (void)id;
  (void)memory;
  (void)size;
  return ge::SUCCESS;
}

Status Session::SetGraphFixedFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  (void)graph_id;
  (void)memory;
  (void)size;
  return ge::SUCCESS;
}

Status Session::UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  (void)graph_id;
  (void)memory;
  (void)size;
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

extern "C" {
ge::Status GeSessionLoadGraph(ge::Session &session, uint32_t graph_id,
                              const std::map<ge::AscendString, ge::AscendString> &option, void *stream) {
  std::cerr << "[STUB] GeSessionLoadGraph, graph id is" << graph_id << std::endl;
  return ge::SUCCESS;
}

ge::graphStatus manager(gert::TensorAddress addr, gert::TensorOperateType operate_type, void **out) {
  if (operate_type == gert::kGetTensorAddress) {
    *out = addr;
  }
  return ge::GRAPH_SUCCESS;
}

ge::Status GeSessionExecuteGraphWithStreamAsync(ge::Session &session, uint32_t graph_id, void *stream,
                                                const std::vector<gert::Tensor> &inputs,
                                                std::vector<gert::Tensor> &outputs) {
  std::cerr << "[STUB] GeSessionExecuteGraphWithStreamAsync, graph id is " << graph_id << std::endl;
  if (inputs.size() < 1U) {
    std::cerr << "[STUB] Input size is empty for graph" << graph_id << std::endl;
    return ge::SUCCESS;
  }
  std::cerr << "[STUB] Input size is " << inputs.size() << std::endl;

  auto placement = gert::TensorPlacement::kOnHost;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].GetPlacement() == gert::TensorPlacement::kOnDeviceHbm) {
      placement = gert::TensorPlacement::kOnDeviceHbm;
      break;
    }
  }
  std::cerr << "[STUB] Output placement is " << placement << std::endl;

  auto spec = ge::GraphSpecManager::GetSpec(graph_id);
  if (outputs.size() != spec.output_dtypes_.size()) {
    std::cerr << "[STUB] Output size " << outputs.size() << " is incompatible with expected "
              << spec.output_dtypes_.size() << std::endl;
    return ge::FAILED;
  }
  std::cerr << "[STUB] Output size is " << outputs.size() << std::endl;

  for (size_t i = 0; i < spec.output_dtypes_.size(); ++i) {
    gert::Tensor &output_i = outputs[i];
    output_i.SetDataType(spec.output_dtypes_[i]);
    output_i.SetPlacement(placement);
    output_i.SetOriginFormat(ge::FORMAT_ND);
    output_i.SetStorageFormat(ge::FORMAT_ND);

    char *st_muti_gear_stub = std::getenv("ST_GEARS_STUB_OUTPUTSHAPE");
    char *st_output_reuse_input_addr_stub = std::getenv("ST_OUTPUT_REUSE_INPUT_ADDR");
    std::vector<int64_t> dims;
    if (st_muti_gear_stub != NULL) {
      dims.resize(inputs[0].GetShape().GetOriginShape().GetDimNum());
      for (size_t i = 0U; i < dims.size(); ++i) {
        dims[i] = inputs[0].GetShape().GetOriginShape().GetDim(i);
      }
    } else {
      dims = spec.netoutput_shapes_[i].GetDims();
    }
    output_i.MutableOriginShape().SetDimNum(dims.size());
    output_i.MutableStorageShape().SetDimNum(dims.size());
    for (size_t j = 0; j < dims.size(); j++) {
      output_i.MutableOriginShape().SetDim(j, dims[j]);
      output_i.MutableStorageShape().SetDim(j, dims[j]);
    }

    static std::vector<float> datas;
    datas.resize(kOutptSize, 0.0);
    gert::TensorAddrManager mgr = nullptr;
    if (placement == gert::TensorPlacement::kOnDeviceHbm) {
      mgr = &manager;
    }
    if (i == 0 && st_output_reuse_input_addr_stub != NULL) {
      std::cerr << "[STUB] Output 0 reuse input 0 addr, and set manager is nullptr. " << std::endl;
      // 模拟复用输入内存的场景
      output_i.SetData(gert::TensorData(inputs[0].GetTensorData().GetAddr(), nullptr,
                                        inputs[0].GetSize(), placement));
    } else {
      output_i.SetData(gert::TensorData(datas.data(), mgr, sizeof(float) * datas.size(),
                                        placement));
    }
  }

  return ge::SUCCESS;
}
} // extern "C"

