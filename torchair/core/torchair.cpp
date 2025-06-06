#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <ATen/record_function.h>
#include "torch/csrc/Exceptions.h"
#include "torchair.h"
#include "hdc_channel.h"

namespace py = pybind11;

namespace {
tng::Status ParseListOptionalTensors(PyObject *obj, std::vector<c10::optional<at::Tensor>> &tensors) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return tng::Status::Error("not a list or tuple");
  }
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (long idx = 0; idx < size; idx++) {
    PyObject *iobj = tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (iobj == Py_None) {
      tensors.emplace_back();
      continue;
    }
    if (!THPVariable_CheckExact(iobj)) {
      return tng::Status::Error("element %l is not a Tensor", idx);
    }
    tensors.emplace_back(THPVariable_Unpack(iobj));
  }
  return tng::Status::Success();
}

tng::Status ParseStream(PyObject *obj, void *&stream) {
  if (obj == Py_None) {
    stream = nullptr;
    return tng::Status::Success();
  }
  if (!THPStream_Check(obj)) {
    return tng::Status::Error("expected Stream object. Got '%s'", Py_TYPE(obj)->tp_name);
  }
  auto c10_stream = c10::Stream::unpack3(((THPStream *)obj)->stream_id, ((THPStream *)obj)->device_index,
                                         static_cast<c10::DeviceType>(((THPStream *)obj)->device_type));
  // TODO: Get acl stream from c10::Stream
  return tng::Status::Success();
}

tng::Status ParseListTensors(PyObject *obj, std::vector<at::Tensor> &tensors) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return tng::Status::Error("not a list or tuple");
  }
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  for (long idx = 0; idx < size; idx++) {
    PyObject *iobj = tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (iobj == Py_None) {
      return tng::Status::Error("element %l is None", idx);
    }
    if (!THPVariable_CheckExact(iobj)) {
      return tng::Status::Error("element %l is not a Tensor", idx);
    }
    tensors.emplace_back(THPVariable_Unpack(iobj));
  }
  return tng::Status::Success();
}

tng::Status ParseListTensors(PyObject *obj, std::vector<const at::Tensor*> &tensors) {
  auto tuple = six::isTuple(obj);
  if (!(tuple || PyList_Check(obj))) {
    return tng::Status::Error("not a list or tuple");
  }
  const auto size = tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  tensors.reserve(size);
  for (long idx = 0; idx < size; idx++) {
    PyObject *iobj = tuple ? PyTuple_GET_ITEM(obj, idx) : PyList_GET_ITEM(obj, idx);
    if (iobj == Py_None) {
      return tng::Status::Error("element %l is None", idx);
    }
    if (!THPVariable_CheckExact(iobj)) {
      return tng::Status::Error("element %l is not a Tensor", idx);
    }
    tensors.emplace_back(&(THPVariable_Unpack(iobj)));
  }
  return tng::Status::Success();
}
}  // namespace

struct TngRuntimeError : public torch::PyTorchError {
  using PyTorchError::PyTorchError;
  PyObject *python_type() override {
    return PyExc_RuntimeError;
  }
};

#define TNG_RAISE_IF_ERROR(expr)                       \
  do {                                                 \
    const auto &status = (expr);                       \
    if (!status.IsSuccess()) {                         \
      throw TngRuntimeError(status.GetErrorMessage()); \
    }                                                  \
  } while (false)

#define TNG_RAISE_ASSERT(expr, msg) \
  do {                              \
    const auto &status = (expr);    \
    if (!status) {                  \
      throw TngRuntimeError(msg);   \
    }                               \
  } while (false)

namespace tng {
void Export(const std::string &serialized_proto, const std::map<std::string, std::string> &options) {
  RECORD_FUNCTION("torchair::Export", {});
  const pybind11::gil_scoped_release release;
  std::map<ge::AscendString, ge::AscendString> compat_options;
  for (const auto &option : options) {
    compat_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
  }
  TNG_RAISE_IF_ERROR(tng::ep::Export(serialized_proto.c_str(), serialized_proto.size(), compat_options));
  const pybind11::gil_scoped_acquire acquire;
}

TorchNpuGraphBase::TorchNpuGraphBase(const std::string &name) : name_(name), concrete_graph_(nullptr){};

void TorchNpuGraphBase::Load(const std::string &serialized_proto, const std::map<std::string, std::string> &options,
                             std::vector<int64_t> input_placements, std::vector<int64_t> output_dtypes,
                             int64_t executor_type) {
  RECORD_FUNCTION("TorchNpuGraphBase::Load", {});
  const pybind11::gil_scoped_release release;
  std::map<ge::AscendString, ge::AscendString> compat_options;
  for (const auto &option : options) {
    compat_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
  }
  TNG_RAISE_IF_ERROR(tng::NpuConcreteGraph::Create(serialized_proto.c_str(), serialized_proto.size(), compat_options,
                                                   input_placements, output_dtypes, executor_type, concrete_graph_));
  const pybind11::gil_scoped_acquire acquire;
}

void TorchNpuGraphBase::SetHintShape(std::vector<std::vector<int64_t>> inputs_shape,
                                     std::vector<std::vector<int64_t>> outputs_shape) {
  RECORD_FUNCTION("TorchNpuGraphBase::SetHintShape", {});
  const pybind11::gil_scoped_release release;
  TNG_RAISE_ASSERT(concrete_graph_, "Set graph inputs and outputs shape must after load");
  TNG_RAISE_IF_ERROR(concrete_graph_->SetHintShape(inputs_shape, outputs_shape));
  const pybind11::gil_scoped_acquire acquire;
}

void TorchNpuGraphBase::Compile() {
  RECORD_FUNCTION("TorchNpuGraphBase::Compile", {});
  const pybind11::gil_scoped_release release;
  TNG_RAISE_IF_ERROR(concrete_graph_->Compile());
  const pybind11::gil_scoped_acquire acquire;
}

void TorchNpuGraphBase::AutoTune(py::object obj) {
  RECORD_FUNCTION("TorchNpuGraphBase::AutoTune", {});
  py::handle handle = obj.cast<py::handle>();
  PyObject *args = handle.ptr();

  PyObject *example_inputs = nullptr;
  PyObject *c10_stream = nullptr;
  TNG_RAISE_ASSERT(PyArg_ParseTuple(args, "OO", &example_inputs, &c10_stream),
                   "Parse arg with signature AutoTune(TensorList example_inputs, c10::Stream) failed");

  void *stream = nullptr;
  TNG_RAISE_IF_ERROR(ParseStream(c10_stream, stream));

  std::vector<at::Tensor> input_tensors;
  TNG_RAISE_IF_ERROR(ParseListTensors(example_inputs, input_tensors));

  const pybind11::gil_scoped_release release;
  TNG_RAISE_IF_ERROR(concrete_graph_->AutoTune(input_tensors, stream));
  const pybind11::gil_scoped_acquire acquire;
}

py::object TorchNpuGraphBase::Run(py::object obj) {
  auto t_begin = tng::GetTimestampForEventLog();
  RECORD_FUNCTION("TorchNpuGraphBase::Run", {});
  PyObject *inputs = nullptr;
  PyObject *assigned_outputs = nullptr;
  PyObject *stream = nullptr;
  py::handle handle = obj.cast<py::handle>();
  PyObject *args = handle.ptr();
  TNG_RAISE_ASSERT(
      PyArg_ParseTuple(args, "OOO", &inputs, &assigned_outputs, &stream),
      "Parse arg with signature Run(TensorList inputs, c10::List<c10::optional<Tensor>> outputs, c10::Stream) failed");

  std::vector<const at::Tensor*> input_tensors;
  // use tensor pointers to reduce movement
  TNG_RAISE_IF_ERROR(ParseListTensors(inputs, input_tensors));
  TNG_RAISE_IF_ERROR(concrete_graph_->AssembleInputs(input_tensors));

  std::vector<c10::optional<at::Tensor>> output_optional_tensors;
  TNG_RAISE_IF_ERROR(ParseListOptionalTensors(assigned_outputs, output_optional_tensors));

  const pybind11::gil_scoped_release release;

  auto t_param = tng::GetTimestampForEventLog();
  std::vector<at::Tensor> outputs;
  TNG_RAISE_IF_ERROR(concrete_graph_->Run(output_optional_tensors, outputs, nullptr));
  auto t_run = tng::GetTimestampForEventLog();

  TNG_LOG(EVENT) << "Graph run at " << t_begin << ", handle param: " << t_param - t_begin
      << "us, input size: " << input_tensors.size() << ", graph run: " << t_run - t_param << "us";

  const pybind11::gil_scoped_acquire acquire;
  return py::cast(outputs);
}

std::string TorchNpuGraphBase::Summary() const {
  return "";
}

void TorchNpuGraphBase::InitializeGraphEngine(const std::map<std::string, std::string> &options) {
  const pybind11::gil_scoped_release release;
  TNG_RAISE_IF_ERROR(tng::NpuConcreteGraph::InitializeResource(options));
  const pybind11::gil_scoped_acquire acquire;
}

void TorchNpuGraphBase::FinalizeGraphEngine() {
  const pybind11::gil_scoped_release release;
  TNG_RAISE_IF_ERROR(tng::NpuConcreteGraph::ReleaseResource());
  const pybind11::gil_scoped_acquire acquire;
}

bool TorchNpuGraphBase::CheckAclnnAvaliable(const std::string &aclnn_name) {
  const pybind11::gil_scoped_release release;
  return tng::NpuOpUtilsTools::CheckAclnnAvaliable(aclnn_name);
}

namespace wrapper {
void StartStdoutChannel(int32_t device) {
  const pybind11::gil_scoped_release release;
  TNG_RAISE_IF_ERROR(tng::StartStdoutChannel(device));
  const pybind11::gil_scoped_acquire acquire;
}
}  // namespace wrapper
}  // namespace tng
