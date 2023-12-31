#ifndef TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_
#define TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Python.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include "torch/torch.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"

#include "concrete_graph.h"

namespace tng {
void Export(const std::string &serialized_proto, const std::map<std::string, std::string> &options);

class TorchNpuGraphBase {
 public:
  explicit TorchNpuGraphBase(const std::string &name);
  ~TorchNpuGraphBase() = default;

  void Load(const std::string &serialized_proto, const std::map<std::string, std::string> &options);

  void Compile();

  void AutoTune(py::object obj);

  py::object Run(py::object obj);

  std::string Summary() const;

  static void InitializeGraphEngine(const std::map<std::string, std::string> &options);

  static void FinalizeGraphEngine();

 private:
  std::string name_;
  std::unique_ptr<tng::NpuConcreteGraph> concrete_graph_ = nullptr;
};
}  // namespace tng

#endif  // TORCH_AIR_TORCH_AIR_CORE_TORCH_AIR_H_