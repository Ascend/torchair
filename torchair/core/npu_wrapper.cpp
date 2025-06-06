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
#include "llm_datadist/llm_datadist.h"
#include "cann_ir_ability.h"

namespace py = pybind11;
namespace npu {
PYBIND11_MODULE(_torchair, m) {
  (void)m.def("StupidRepeat", [](const char *device_name, int times) {
    for (int i = 0; i < times; i++) {
      std::cerr << device_name;
    }
  });

  (void)m.def("InitializeGraphEngine", &tng::TorchNpuGraphBase::InitializeGraphEngine);

  (void)m.def("InitDeviceStdoutChannel", &tng::wrapper::StartStdoutChannel);

  (void)m.def("FinalizeGraphEngine", &tng::TorchNpuGraphBase::FinalizeGraphEngine);

  (void)m.def("export", &tng::Export);

  (void)m.def("CheckAclnnAvaliable", &tng::TorchNpuGraphBase::CheckAclnnAvaliable);

  py::class_<tng::TorchNpuGraphBase>(m, "TorchNpuGraphBase")
    .def(py::init<const std::string &>())
    .def("load", &tng::TorchNpuGraphBase::Load)
    .def("set_hint_shape", &tng::TorchNpuGraphBase::SetHintShape)
    .def("compile", &tng::TorchNpuGraphBase::Compile)
    .def("auto_tune", &tng::TorchNpuGraphBase::AutoTune)
    .def("summary", &tng::TorchNpuGraphBase::Summary)
    .def("run", &tng::TorchNpuGraphBase::Run);

  (void)py::enum_<llm_datadist::TorchDataType>(m, "TorchDataType")
      .value("BOOL", llm_datadist::TorchDataType::kBool)
      .value("UINT8", llm_datadist::TorchDataType::kUint8)
      .value("INT8", llm_datadist::TorchDataType::kInt8)
      .value("INT16", llm_datadist::TorchDataType::kInt16)
      .value("INT32", llm_datadist::TorchDataType::kInt32)
      .value("INT64", llm_datadist::TorchDataType::kInt64)
      .value("BF16", llm_datadist::TorchDataType::kBfloat16)
      .value("FLOAT16", llm_datadist::TorchDataType::kFloat16)
      .value("FLOAT32", llm_datadist::TorchDataType::kFloat32)
      .value("FLOAT64", llm_datadist::TorchDataType::kFloat64)
      .value("COMPLEX32", llm_datadist::TorchDataType::kComplex32)
      .value("COMPLEX64", llm_datadist::TorchDataType::kComplex64)
      .value("COMPLEX128", llm_datadist::TorchDataType::kComplex128)
      .export_values();
  (void)m.def("as_torch_tensors", &llm_datadist::AsTorchTensor);
  m.def("check_cann_compat", &cann_ir_ability::CheckCannCompat);
};
}  // namespace npu
