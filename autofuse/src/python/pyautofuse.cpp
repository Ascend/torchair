#include <Python.h>

#include "optimize.h"
#include "codegen.h"

#include "pyascir.h"
#include "pyascir_types.h"

namespace pyascir {
class Autofuser {
 public:
  struct Object {
    PyObject_HEAD

    optimize::Optimizer* optimizer;
    codegen::Codegen* codegen;
  };

  static PyTypeObject Type;
  static PyMethodDef Methods[];

  static void _dealloc(PyObject *self);
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
  static int _init(PyObject *self, PyObject *args, PyObject *kwds);

  static PyObject* autofuse(PyObject *self, PyObject *args, PyObject *kwds);
  static PyObject* codegen(PyObject *self, PyObject *args, PyObject *kwds);
};

PyMethodDef Autofuser::Methods[] = {
    {"autofuse", (PyCFunction)Autofuser::autofuse, METH_VARARGS | METH_KEYWORDS, NULL},
    {"codegen", (PyCFunction)Autofuser::codegen, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}
};

PyTypeObject Autofuser::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Autofuser",
    .tp_basicsize = sizeof(Autofuser::Object),
    .tp_itemsize = 0,
    .tp_dealloc = Autofuser::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Autofuser",
    .tp_methods = Autofuser::Methods,
    .tp_members = nullptr,
    .tp_init = nullptr,
    .tp_new = Autofuser::_new
};

void Autofuser::_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *Autofuser::_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  auto self = (Autofuser::Object*)type->tp_alloc(type, 0);
  if (self == nullptr) {
    return nullptr;
  }

  self->optimizer = nullptr;
  self->codegen = nullptr;

  return (PyObject *)self;
}

int Autofuser::_init(PyObject *self, PyObject *args, PyObject *kwds) {
  auto self_ = (Autofuser::Object *)self;

  self_->optimizer = new optimize::Optimizer(optimize::OptimizerOptions{});
  if (self_->optimizer == nullptr) {
    return -1;
  }

  self_->codegen = new codegen::Codegen(codegen::CodegenOptions{});
  if (self_->codegen == nullptr) {
    delete self_->optimizer;
    return -1;
  };

  return 0;
}

PyObject *Autofuser::autofuse(PyObject *self, PyObject *args, PyObject *kwds) {
  auto self_ = (Autofuser::Object *)self;

  pyascir::HintGraph::Object* hint_graph = nullptr;
  if (!PyArg_ParseTuple(args, "O", &hint_graph)) {
      return PyErr_Format(PyExc_ValueError, "autofuse requires a hint graph");
  }

  if (!PyObject_IsInstance((PyObject*)hint_graph, (PyObject*)&pyascir::HintGraph::Type)) {
      return PyErr_Format(PyExc_ValueError, "autofuse requires a hint graph");
  }

  std::string optimize_graph_name = hint_graph->graph->GetName() + "_optimized";
  ascir::ImplGraph optimize_graph(optimize_graph_name.c_str());
  optimize_graph.CopyFrom(*hint_graph->graph);

  int ret = self_->optimizer->GetApiInfo(*hint_graph->graph, optimize_graph);
  if (ret != 0) {
    return PyErr_Format(PyExc_ValueError, "autofuse get api info failed");
  }

  std::vector<ascir::ImplGraph> impl_graphs;
  ret = self_->optimizer->AutoScheduler(*hint_graph->graph, optimize_graph, impl_graphs);
  if (ret != 0) {
    return PyErr_Format(PyExc_ValueError, "autofuse auto scheduler failed");
  }

  ret = self_->optimizer->BufQueAlloc(*hint_graph->graph, impl_graphs);
  if (ret != 0) {
    return PyErr_Format(PyExc_ValueError, "autofuse buf que alloc failed");
  }

  PyObject* ret_graphs = PyList_New(0);
  for (auto &impl_graph : impl_graphs) {
    PyList_Append(ret_graphs, pyascir::HintGraph::FromGraph(&impl_graph));
  }

  return ret_graphs;
}

PyObject *Autofuser::codegen(PyObject *self, PyObject *args, PyObject *kwds) {
  Py_RETURN_NONE;
}
}

static PyModuleDef PyAutofuseModule = {
    PyModuleDef_HEAD_INIT,
    "autofuse",
    "Autofuse module",
    -1,
};

PyMODINIT_FUNC PyInit_pyautofuse(void) {
  auto pyautofuse_module = PyModule_Create(&PyAutofuseModule);
  if (pyautofuse_module == NULL) {
    return NULL;
  }

  auto pyascir_module = PyInit_ascir();
  if (pyascir_module == nullptr) {
    Py_DECREF(pyautofuse_module);
    return NULL;
  }

  PyModule_AddObject(pyautofuse_module, "ascir", pyascir_module);

  if (PyType_Ready(&pyascir::Autofuser::Type) < 0) {
    Py_DECREF(pyautofuse_module);
    Py_DECREF(pyascir_module);
    return nullptr;
  }
  PyModule_AddObject(pyautofuse_module, "Autofuser", (PyObject*)&pyascir::Autofuser::Type);

  return pyautofuse_module;
}
