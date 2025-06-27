#include <cstdarg>

#include "concrete_graph.h"

#include "graph/types.h"
#include "ge/ge_api.h"

#include "checker.h"
#include "compat_apis.h"
#include "executor.h"
#include "graph_data.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "logger.h"
#include "session.h"
#include "utils.h"
#include "acl/acl_op_compiler.h"

char *CreateMessage(const char *format, va_list arg);

namespace {
tng::Status GetAclCompileopt(aclCompileOpt opt, std::string& val, std::string optName) {
  auto opt_size = aclGetCompileoptSize(opt);
  if (opt_size != 0UL) {
    char value[opt_size];
    auto acl_ret = aclGetCompileopt(opt, value, opt_size);
    if (acl_ret == ACL_ERROR_API_NOT_SUPPORT) {
      TNG_LOG(WARNING) << "ACL get compile opt, " << optName << " unsupport, opt size " << opt_size;
      return tng::Status::Success();
    }
    TNG_ASSERT(acl_ret == ACL_SUCCESS, "ACL get compile opt failed, return %d", acl_ret);
    val = std::string(value);
  }
  return tng::Status::Success();
}

tng::Status NormalizeCompileOptions(const std::map<ge::AscendString, ge::AscendString> &options,
                                    std::map<ge::AscendString, ge::AscendString> &normalized_options) {
  normalized_options = options;

  // Use separate memory cleaning for atomic nodes to better reuse memory.
  // (void)normalized_options.insert(std::make_pair(ge::ATOMIC_CLEAN_POLICY.c_str(), "1"));

  (void)normalized_options.insert(std::make_pair(ge::MEMORY_OPTIMIZATION_POLICY.c_str(), "MemoryPriority"));

  std::string val = "";
  TNG_RETURN_IF_ERROR(GetAclCompileopt(ACL_OP_DEBUG_OPTION, val, "ACL_OP_DEBUG_OPTION"));
  if (!val.empty()) {
    (void)normalized_options.insert(std::make_pair("op_debug_option", val.c_str()));
  }

  return tng::Status::Success();
}

std::vector<tng::Placement> ParsePlacements(const std::vector<int64_t> &input_placements) {
  std::vector<tng::Placement> placements;
  placements.reserve(input_placements.size());
  for (const auto &placement : input_placements) {
    if (placement == 0) {
      placements.push_back(tng::Placement::HOST);
    } else if (placement == 1) {
      placements.push_back(tng::Placement::DEVICE);
    } else {
      placements.push_back(tng::Placement::UNKNOWN);
    }
  }
  return placements;
}

std::vector<ge::DataType> ParseOutputDtypes(const std::vector<int64_t> &output_dtypes) {
  std::vector<ge::DataType> dtypes;
  dtypes.reserve(output_dtypes.size());
  for (const auto &dtype : output_dtypes) {
    if (dtype < 0 || dtype >= static_cast<int64_t>(ge::DataType::DT_MAX)) {
      dtypes.push_back(ge::DataType::DT_UNDEFINED);
      continue;
    }
    dtypes.push_back(static_cast<ge::DataType>(dtype));
  }
  return dtypes;
}

tng::ExecutorType ParseExecutorType(int64_t executor_type) {
  if (executor_type == 0) {
    return tng::ExecutorType::CPU;
  }
  if (executor_type == 1) {
    return tng::ExecutorType::NPU;
  }
  return tng::ExecutorType::UNKNOWN;
}

tng::Status CheckNetOutputShape(const std::vector<std::vector<int64_t>> &output_shapes, std::vector<ge::Shape> &output_ge_shapes) {
  TNG_LOG(DEBUG) << "FX graph NetOutput shapes is : " << tng::DebugString(output_shapes);
  TNG_LOG(DEBUG) << "--------";
  TNG_LOG(DEBUG) << "Ascend GE graph NetOutput shapes is : " << tng::DebugString(output_ge_shapes);
  if (output_shapes.size() != output_ge_shapes.size()) {
    return tng::Status::Error("The number of Ascend GE graph NetOutput: %d is not equal to FX graph NetOutput: %d. "
        "FX graph NetOutput shapes is : [%s], Ascend GE graph NetOutput shapes is : [%s]", output_ge_shapes.size(), output_shapes.size(),
        tng::DebugString(output_shapes).c_str(), tng::DebugString(output_ge_shapes).c_str());
  }
  for (size_t i = 0u; i < output_shapes.size(); ++i) {
    const auto& out_dim = output_shapes[i];
    const auto& out_ge_dim = output_ge_shapes[i].GetDims();
    if (out_dim.size() != out_ge_dim.size()) {
      TNG_LOG(ERROR) << "The dim size of Ascend GE graph NetOutput: ["<< out_ge_dim <<"] is not equal to FX graph NetOutput: ["<< out_dim <<"]";
      return tng::Status::Error("The dim size of Ascend GE graph NetOutput: %s is not equal to FX graph NetOutput: %s. "
          "FX graph NetOutput shapes is : %s, Ascend GE graph NetOutput shapes is : %s", tng::DebugString(out_ge_dim).c_str(), tng::DebugString(out_dim).c_str(),
          tng::DebugString(output_shapes).c_str(), tng::DebugString(output_ge_shapes).c_str());
    }

    for (size_t j = 0u; j < out_dim.size(); ++j) {
      if (out_dim[j] == -1 || out_ge_dim[j] == -1) {
        TNG_LOG(WARNING) << "The current dim of output_shapes is dynamic shape, can not check shape";
        continue;
      }
      if (out_dim[j] != out_ge_dim[j]) {
        TNG_LOG(ERROR) << "The dim of Ascend GE graph NetOutput: ["<< out_ge_dim <<"] is not equal to FX graph NetOutput: ["<< out_dim <<"]";
        return tng::Status::Error("The dim of Ascend GE graph NetOutput: %s is not equal to FX graph NetOutput: %s. "
            "FX graph NetOutput shapes is : %s, Ascend GE graph NetOutput shapes is : %s", tng::DebugString(out_ge_dim).c_str(), tng::DebugString(out_dim).c_str(),
            tng::DebugString(output_shapes).c_str(), tng::DebugString(output_ge_shapes).c_str());
      }
    }
  }
  return tng::Status::Success();
}

tng::Status CheckNetOutDtypes(const std::vector<ge::DataType> &output_dtypes, const std::vector<ge::DataType> &output_ge_dtypes) {
  TNG_LOG(DEBUG) << "FX graph NetOutput dtypes is : " << tng::DebugString(output_dtypes);
  TNG_LOG(DEBUG) << "Ascend GE graph NetOutput dtypes is : " << tng::DebugString(output_ge_dtypes);
  if (output_dtypes.size() != output_ge_dtypes.size()) {
    return tng::Status::Error("The size of Ascend GE graph NetOutput dtypes: %d is not equal to FX graph NetOutput dtypes: %d. "
        "FX graph NetOutput dtypes is : %s, Ascend GE graph NetOutput dtypes is : %s", output_ge_dtypes.size(), output_dtypes.size(),
        tng::DebugString(output_dtypes).c_str(), tng::DebugString(output_ge_dtypes).c_str());
  }
  for (size_t i = 0u; i < output_dtypes.size(); ++i) {
    if (output_dtypes[i] != output_ge_dtypes[i]) {
      return tng::Status::Error("The dtype in num[%d] net output of Ascend output: [%s] is not equal to FX graph NetOutput: [%s]. "
          "FX graph NetOutput dtypes is : %s, Ascend GE graph NetOutput dtypes is : %s", i, tng::DebugString(output_ge_dtypes[i]).c_str(),
          tng::DebugString(output_dtypes[i]).c_str(), tng::DebugString(output_dtypes).c_str(), tng::DebugString(output_ge_dtypes).c_str());
    }
  }
  return tng::Status::Success();
}

}  // namespace
namespace tng {
NpuConcreteGraph::NpuConcreteGraph(std::shared_ptr<GraphData> graph_data) : graph_data_(std::move(graph_data)){};

Status NpuConcreteGraph::ReleaseResource() {
  return Session::GetInstance().Finalize();
}

Status NpuConcreteGraph::InitializeResource(const std::map<std::string, std::string> &options) {
  return Session::GetInstance().Initialize(options);
}

Status NpuConcreteGraph::Create(const void *serialized_proto, size_t proto_size,
                                const std::map<ge::AscendString, ge::AscendString> &options,
                                std::vector<int64_t> input_placements, std::vector<int64_t> output_dtypes,
                                int64_t executor_type, std::unique_ptr<NpuConcreteGraph> &graph) {
  TNG_LOG(INFO) << "Creating concrete graph from proto with size " << proto_size;
  TNG_ASSERT_NOTNULL(serialized_proto, "Given serialized proto is nullptr.");
  TNG_ASSERT(ge::IntegerChecker<int32_t>::Compat(proto_size), "Proto size %zu exceed 2G limit.", proto_size);

  auto graph_data = std::make_unique<GraphData>();
  TNG_RETURN_IF_ERROR(compat::ParseGraphFromArray(serialized_proto, proto_size, graph_data->graph));

  graph_data->input_placements = ParsePlacements(input_placements);
  graph_data->output_dtypes = ParseOutputDtypes(output_dtypes);
  graph_data->executor_type = ParseExecutorType(executor_type);
  TNG_ASSERT(graph_data->executor_type != ExecutorType::UNKNOWN, "Executor type is unknown.");

  TNG_RETURN_IF_ERROR(NormalizeCompileOptions(options, graph_data->compile_options));
  if (graph_data->executor_type == ExecutorType::NPU) {
    // Use zero copy on inputs and outputs to reuse user memory only for npu executor.
    (void)graph_data->compile_options.insert(std::make_pair(ge::OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1"));
  }
  const char *const deterministic_option = "ge.deterministic";
  auto iter = graph_data->compile_options.find(ge::AscendString(deterministic_option));
  if (iter != graph_data->compile_options.end()) {
    graph_data->deterministic_value = (iter->second == static_cast<ge::AscendString>("1")) ? 1 : 0;
  }
  const char *const frozenInput_option = "frozenInput";
  auto frozen_iter = graph_data->compile_options.find(ge::AscendString(frozenInput_option));
  if (frozen_iter != graph_data->compile_options.end()) {
    graph_data->frozen_input_flag_list = Split(frozen_iter->second.GetString(), ',');
  }
  if (graph_data->frozen_input_flag_list.empty()) {
    graph_data->frozen_input_flag_list.resize(input_placements.size(), 0);
  }

  static std::atomic_uint32_t uuid{0U};
  graph_data->id = uuid++;

  TNG_LOG(INFO) << DebugString(*graph_data);

  graph.reset(new NpuConcreteGraph(std::move(graph_data)));
  TNG_ASSERT_NOTNULL(graph, "Failed to create graph.");

  TNG_LOG(INFO) << "Concrete graph from proto with size " << proto_size << " created.";

  return Status::Success();
}

Status NpuConcreteGraph::Compile() {
  TNG_LOG(INFO) << "Compiling concrete graph " << graph_data_->id << " with options:";
  for (const auto &option : graph_data_->compile_options) {
    TNG_LOG(INFO) << "    " << option.first.GetString() << " = " << option.second.GetString();
  }

  TNG_RETURN_IF_ERROR(
      Session::GetInstance().AddGraph(graph_data_->id, *graph_data_->graph, graph_data_->compile_options));
  is_graph_added_ = true;
  graph_id_ = graph_data_->id;
  created_pid_ = getpid();

  if (graph_data_->executor_type == ExecutorType::NPU) {
    // Only device input is supported for compile
    TNG_RETURN_IF_ERROR(Session::GetInstance().CompileGraph(graph_data_->id, graph_data_->summary));
    if (graph_data_->summary->IsStatic()) {      
      // Check the shape and dtype of Ascend net output is same as FX net output
      std::vector<ge::Shape> output_ge_shapes;
      TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputShapes(output_ge_shapes));
      TNG_RETURN_IF_ERROR(CheckNetOutputShape(graph_data_->outputs_shape, output_ge_shapes));

      std::vector<ge::DataType> output_ge_dtypes;
      TNG_ASSERT_GE_OK(graph_data_->summary->GetOutputDtypes(output_ge_dtypes));
      TNG_RETURN_IF_ERROR(CheckNetOutDtypes(graph_data_->output_dtypes, output_ge_dtypes));
    }
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

Status NpuConcreteGraph::Run(const std::vector<c10::optional<at::Tensor>> &torch_outputs,
                             std::vector<at::Tensor> &outputs, void *stream) {
  TNG_LOG(INFO) << "Run concrete graph " << graph_data_->id << " with stream " << stream;

  HcclConfigValue hccl_config = {graph_data_->deterministic_value};
  TNG_ASSERT(HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, hccl_config) == HCCL_SUCCESS,
             "Failed to set HCCL_DETERMINISTIC.");
  TNG_ASSERT_NOTNULL(executor_, "Executor is not initialized.");
  TNG_RETURN_IF_ERROR(executor_->Run(torch_outputs, outputs, stream));
  return Status::Success();
}

Status NpuConcreteGraph::SetHintShape(const std::vector<std::vector<int64_t>> &inputs_shape,
                                      const std::vector<std::vector<int64_t>> &outputs_shape) {
  TNG_ASSERT(graph_data_, "After load graph, graph_data_ should not nullptr.");
  graph_data_->inputs_shape = inputs_shape;
  graph_data_->outputs_shape = outputs_shape;
  return Status::Success();
}

Status NpuConcreteGraph::AssembleInputs(const std::vector<const at::Tensor*> &tensors) {
  TNG_RETURN_IF_ERROR(executor_->AssembleInputs(tensors));
  return Status::Success();
}
}  // namespace tng
