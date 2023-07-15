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

namespace py = pybind11;
namespace npu {
PYBIND11_MODULE(_npu_graph_executor, m){};
}  // namespace npu
