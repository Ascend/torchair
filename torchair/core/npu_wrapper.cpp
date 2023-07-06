/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

  py::class_<tng::TorchNpuGraphBase>(m, "TorchNpuGraphBase")
    .def(py::init<const std::string &>())
    .def("load", &tng::TorchNpuGraphBase::Load)
    .def("compile", &tng::TorchNpuGraphBase::Compile)
    .def("auto_tune", &tng::TorchNpuGraphBase::AutoTune)
    .def("summary", &tng::TorchNpuGraphBase::Summary)
    .def("run", &tng::TorchNpuGraphBase::Run);
};
}  // namespace npu
