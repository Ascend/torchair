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

#include "torchair.h"

namespace py = pybind11;
namespace npu {
PYBIND11_MODULE(_torchair, m) {
  (void)m.def("StupidRepeat", [](const char *device_name, int times) {
    for (int i = 0; i < times; i++) {
      std::cerr << device_name;
    }
  });

  (void)m.def("InitializeGraphEngine", &tng::TorchNpuGraphBase::InitializeGraphEngine);

  (void)m.def("FinalizeGraphEngine", &tng::TorchNpuGraphBase::FinalizeGraphEngine);

  (void)m.def("export", &tng::Export);

  py::class_<tng::TorchNpuGraphBase>(m, "TorchNpuGraphBase")
    .def(py::init<const std::string &>())
    .def("load", &tng::TorchNpuGraphBase::Load)
    .def("set_hint_shape", &tng::TorchNpuGraphBase::SetHintShape)
    .def("compile", &tng::TorchNpuGraphBase::Compile)
    .def("auto_tune", &tng::TorchNpuGraphBase::AutoTune)
    .def("summary", &tng::TorchNpuGraphBase::Summary)
    .def("run", &tng::TorchNpuGraphBase::Run);
};
}  // namespace npu
