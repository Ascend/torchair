#ifndef TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_
#define TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/torch.h"

#include "concrete_graph.h"
#include "utils_tools.h"
#include "logger.h"

namespace tng {
void Export(const std::string &serialized_proto, const std::map<std::string, std::string> &options);

class TorchNpuGraphBase {
 public:
  explicit TorchNpuGraphBase(const std::string &name);
  ~TorchNpuGraphBase() = default;

  void Load(const std::string &serialized_proto, const std::map<std::string, std::string> &options,
            std::vector<int64_t> input_placements, std::vector<int64_t> output_dtypes, int64_t executor_type);

  void Compile();

  void AutoTune(py::object obj);

  void SetHintShape(std::vector<std::vector<int64_t>> inputs_shape,
                    std::vector<std::vector<int64_t>> outputs_shape);

  py::object Run(py::object obj);

  std::string Summary() const;

  static void InitializeGraphEngine(const std::map<std::string, std::string> &options);

  static void FinalizeGraphEngine();

  static bool CheckAclnnAvaliable(const std::string &aclnn_name);

 private:
  std::string name_;
  std::unique_ptr<tng::NpuConcreteGraph> concrete_graph_ = nullptr;
};
namespace wrapper {
void StartStdoutChannel(int32_t device);
}  // namespace wrapper
}  // namespace tng
#endif  // TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_