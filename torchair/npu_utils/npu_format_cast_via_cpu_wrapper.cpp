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

#include "npu_format_cast_via_cpu.h"

namespace tng {

namespace py = pybind11;
namespace npu_utils {
PYBIND11_MODULE(_npu_utils, m) {

  m.def("NpuFormatCastViaCpu", &npu_format_cast_via_cpu,
        py::arg("self"),
        py::arg("acl_format"),
        py::arg("customize_dtype") = py::none(),
	    py::arg("input_dtype") = py::none());
};
}  // namespace npu_utils
}

