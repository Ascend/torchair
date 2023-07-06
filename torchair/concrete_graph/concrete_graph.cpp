#include <cstdarg>

#include "concrete_graph.h"

#include "external/graph/types.h"
#include "ge/ge_api.h"
#include "ge_ir.pb.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/detail/model_serialize_imp.h"

#include "checker.h"
#include "session.h"
#include "graph_data.h"
#include "executor.h"
#include "logger.h"
#include "utils.h"
#include "compat_apis.h"

char *CreateMessage(const char *format, va_list arg);

namespace {
tng::Status NormalizeCompileOptions(const std::map<ge::AscendString, ge::AscendString> &options,
                                    std::map<ge::AscendString, ge::AscendString> &normalized_options) {
  normalized_options = options;
  normalized_options[ge::JIT_COMPILE.c_str()] = "2";
  auto iter = options.find("jit_compile");
  if (iter != options.end()) {
    normalized_options.erase("jit_compile");
    if (iter->second == ge::AscendString("0")) {
      normalized_options[ge::JIT_COMPILE.c_str()] = "0";
    }
  }

  return tng::Status::Success();
}
}  // namespace
namespace tng {
NpuConcreteGraph::NpuConcreteGraph(std::shared_ptr<GraphData> graph_data) : graph_data_(std::move(graph_data)){};

Status NpuConcreteGraph::ReleaseResource() { return Session::GetInstance().Finalize(); }

Status NpuConcreteGraph::InitializeResource(const std::map<std::string, std::string> &options) {
  return Session::GetInstance().Initialize(options);
}

Status NpuConcreteGraph::Create(const void *serialized_proto, size_t proto_size,
                                const std::map<ge::AscendString, ge::AscendString> &options,
                                std::unique_ptr<NpuConcreteGraph> &graph) {
  TNG_LOG(INFO) << "Creating concrete graph from proto with size " << proto_size;
  TNG_ASSERT_NOTNULL(serialized_proto, "Given serialized proto is nullptr");
  TNG_ASSERT(ge::IntegerChecker<int32_t>::Compat(proto_size), "Proto size %zu exceed 2G limit", proto_size);

  auto graph_data = std::make_unique<GraphData>();

  TNG_ASSERT(graph_data->graph_def.ParseFromArray(serialized_proto, proto_size));
  TNG_LOG(INFO) << "Graph parsed successfully and " << graph_data->graph_def.op_size() << " ops parsed";

  TNG_RETURN_IF_ERROR(compat::ConvertGraphDefToGraph(graph_data->graph_def, graph_data->graph));

  graph_data->input_placements = compat::GetGraphInputPlacemnts(graph_data->graph_def);
  graph_data->output_dtypes = compat::GetGraphOutputDtypes(graph_data->graph_def);

  TNG_RETURN_IF_ERROR(NormalizeCompileOptions(options, graph_data->compile_options));

  static std::atomic_uint32_t uuid = 0U;
  graph_data->id = uuid++;

  TNG_LOG(INFO) << DebugString(*graph_data);

  graph.reset(new NpuConcreteGraph(std::move(graph_data)));
  TNG_ASSERT_NOTNULL(graph, "Failed to create graph");

  TNG_LOG(INFO) << "Concrete graph from proto with size " << proto_size << " created";

  return Status::Success();
}

Status NpuConcreteGraph::Compile() {
  TNG_LOG(INFO) << "Compiling concrete graph " << graph_data_->id << " with options:";
  for (const auto &option : graph_data_->compile_options) {
    TNG_LOG(INFO) << "    " << option.first.GetString() << " = " << option.second.GetString();
  }
  TNG_RETURN_IF_ERROR(
    Session::GetInstance().AddGraph(graph_data_->id, *graph_data_->graph, graph_data_->compile_options));
  if (std::find(graph_data_->input_placements.begin(), graph_data_->input_placements.end(), Placement::DEVICE) !=
      graph_data_->input_placements.end()) {
    // Only device input is not supported for compile
    TNG_RETURN_IF_ERROR(Session::GetInstance().CompileGraph(graph_data_->id, &graph_data_->summary));
  }

  TNG_RETURN_IF_ERROR(Executor::Create(graph_data_, executor_));
  return Status::Success();
}

Status NpuConcreteGraph::AutoTune(const std::vector<at::Tensor> &example_inputs, void *stream) {
  TNG_LOG(INFO) << "Auto-tunning concrete graph " << graph_data_->id;
  std::vector<ge::Tensor> inputs;
  inputs.resize(example_inputs.size());
  for (size_t i = 0U; i < inputs.size(); ++i) {
    TNG_RETURN_IF_ERROR(AtTensorToGeTensor(example_inputs[i], inputs[i]));
    TNG_LOG(INFO) << "Assemble aten input " << i << " " << DebugString(example_inputs[i]) << " to "
                  << DebugString(inputs[i]);
  }
  TNG_RETURN_IF_ERROR(
    Session::GetInstance().AutoTuneGraph(*graph_data_->graph, graph_data_->compile_options, inputs, stream));
  return Status::Success();
}

Status NpuConcreteGraph::Run(const std::vector<at::Tensor> &torch_inputs,
                             const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                             std::vector<at::Tensor> &outputs, void *stream) {
  TNG_LOG(INFO) << "Run concrete graph " << graph_data_->id << " with stream " << stream;
  TNG_ASSERT_NOTNULL(executor_, "Executor is not initialized");
  TNG_RETURN_IF_ERROR(executor_->Run(torch_inputs, torch_outputs, outputs, stream));
  return Status::Success();
}
}  // namespace tng