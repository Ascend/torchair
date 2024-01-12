#include "pyascir_types.h"

/** SizeVar */
namespace pyascir {
void SizeVar::_dealloc(PyObject *self_pyobject) {
  auto self = (Object *)self_pyobject;
  Py_XDECREF(self->name);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *SizeVar::_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->id = -1;
    self->value = -1;
    self->name = Py_None;
    self->type = Py_None;
  }
  return (PyObject *)self;
}

int SizeVar::_init(PyObject *self_pyobject, int id, int value, const char *name, const char *type) {
  auto self = (Object *)self_pyobject;
  self->id = id;
  self->value = value;
  self->name = PyUnicode_FromString(name);
  self->type = PyUnicode_FromString(type);
  return 0;
}


int SizeVar::_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  int id;
  int value;
  char *name = nullptr;
  char *type = nullptr;
  if (!PyArg_ParseTuple(args, "iiss", &id, &value, &name, &type)) {
    return -1;
  }

  return _init(self_pyobject, id, value, name, type);
}

PyMemberDef SizeVar::Members[] = {
    {"id", T_INT, offsetof(SizeVar::Object, id), 0, "SizeVar id"},
    {"value", T_INT, offsetof(SizeVar::Object, value), 0, "Size"},
    {"name", T_OBJECT, offsetof(SizeVar::Object, name), 0, "Name"},
    {"type", T_OBJECT, offsetof(SizeVar::Object, type), 0, "Type"},
    {NULL}
};

PyTypeObject SizeVar::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "SizeVar",
    .tp_basicsize = sizeof(SizeVar::Object),
    .tp_itemsize = 0,
    .tp_dealloc = SizeVar::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "SizeVar object",
    .tp_methods = nullptr,
    .tp_members = SizeVar::Members,
    .tp_init = SizeVar::_init,
    .tp_new = SizeVar::_new,
};
} // namespace pyascir

/** SizeExpr */
namespace pyascir {
PyMemberDef SizeExpr::Members[] = {
    {"is_zero", T_OBJECT, offsetof(SizeExpr::Object, is_zero), 0, "is zero"},
    {"nums", T_OBJECT, offsetof(SizeExpr::Object, nums), 0, "List of Size numerators"},
    {"dens", T_OBJECT, offsetof(SizeExpr::Object, dens), 0, "List of size denominators"},
    {NULL}
};

PyTypeObject SizeExpr::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "SizeExpr",
    .tp_basicsize = sizeof(SizeExpr::Object),
    .tp_itemsize = 0,
    .tp_dealloc = SizeExpr::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "SizeExpr object",
    .tp_methods = nullptr,
    .tp_members = SizeExpr::Members,
    .tp_init = SizeExpr::_init,
    .tp_new = SizeExpr::_new,
};

void SizeExpr::_dealloc(PyObject *self) {
  auto self_ = (Object *)self;
  Py_XDECREF(self_->nums);
  Py_XDECREF(self_->dens);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *SizeExpr::_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)type->tp_alloc(type, 0);
  if (self!= nullptr) {
    self->is_zero = PyBool_FromLong(false);
    self->nums = PyList_New(0);
    self->dens = PyList_New(0);
  }
  return (PyObject *)self;
}

int SizeExpr::_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  PyObject *nums = nullptr;
  PyObject *dens = nullptr;
  if (!PyArg_ParseTuple(args, "|OO", &nums, &dens)) {
    return -1;
  }

  auto self = (Object *)self_pyobject;

  // Support construct by ascir.SizeExpr([0])
  if (nums != nullptr && PyList_Check(nums) && PyList_Size(nums) == 1
      && (dens == nullptr || PyList_Check(dens) && PyList_Size(dens) == 0)) {
    auto num = PyList_GetItem(nums, 0);
    if (PyLong_Check(num) && PyLong_AsLong(num) == 0) {
        self->is_zero = PyBool_FromLong(true);
        return 0;
    } else if (PyLong_Check(num) && PyLong_AsLong(num) == 1) {
        return 0;
    }
  }

  // Construct from ascir.SizeExpr([s0, s1], [s2, s3])
  if (nums != nullptr) {
    for (int i = 0; i < PyList_Size(nums); ++i) {
      auto size = (SizeVar::Object*)PyList_GetItem(nums, i);
      if (!PyObject_IsInstance((PyObject*)size, (PyObject*)&SizeVar::Type)) {
        return -1;
      }
      PyList_Append(self->nums, PyLong_FromLong(size->id));
    }
  }
  if (dens != nullptr) {
    for (int i = 0; i < PyList_Size(dens); ++i) {
      auto size = (SizeVar::Object *)PyList_GetItem(dens, i);
      if (!PyObject_IsInstance((PyObject*)size, (PyObject*)&SizeVar::Type)) {
        return -1;
      }
      PyList_Append(self->dens, PyLong_FromLong(size->id));
    }
  }

  return 0;
}

int SizeExpr::_init(PyObject *self_pyobject,
                    const bool is_zero,
                    const std::vector<ascir::SizeVarId> &nums,
                    const std::vector<ascir::SizeVarId> &dens) {
  auto self = (Object *)self_pyobject;
  self->is_zero = PyBool_FromLong(is_zero);
  for (auto num : nums) {
      PyList_Append(self->nums, PyLong_FromLong(num));
  }
  for (auto den : dens) {
      PyList_Append(self->dens, PyLong_FromLong(den));
  }
  return 0;
}

PyObject *SizeExpr::FromSizeExpr(const ascir::SizeExpr &expr) {
  auto size = _new(&SizeExpr::Type, nullptr, nullptr);
  if (size == nullptr) {
    return nullptr;
  }

  _init(size, expr.is_zero, expr.nums, expr.dens);
  Py_IncRef(size);
  return size;
}

ascir::SizeExpr SizeExpr::AsSizeExpr(PyObject *obj) {
  if (PyObject_IsInstance(obj, (PyObject*)&SizeExpr::Type)) {
    ascir::SizeExpr size_expr;

    auto size = (SizeExpr::Object *)obj;

    size_expr.is_zero = size->is_zero == Py_True;
    for (int i = 0; i < PyList_Size(size->nums); ++i) {
      size_expr.nums.push_back(PyLong_AsLong(PyList_GetItem(size->nums, i)));
    }
    for (int i = 0; i < PyList_Size(size->dens); ++i) {
      size_expr.dens.push_back(PyLong_AsLong(PyList_GetItem(size->dens, i)));
    }

    return size_expr;
  } else if (PyLong_Check(obj)) {
    int v = PyLong_AsLong(obj);
    if (v == 0) {
      return ascir::SizeExpr::Zero();
    } else if (v == 1) {
      return ascir::SizeExpr::One();
    } else {
      return ascir::SizeExpr{};
    }
  } else {
    return ascir::SizeExpr{};
  }
}
} // namespace pyascir for SizeExpr

/** Axis */
namespace pyascir {
PyMemberDef Axis::Members[] = {
    {"id", T_INT, offsetof(Axis::Object, id), 0, "Axis id"},
    {"size", T_OBJECT, offsetof(Axis::Object, size), 0, "Size expression"},
    {"name", T_OBJECT, offsetof(Axis::Object, name), 0, "Name"},
    {"type", T_OBJECT, offsetof(Axis::Object, type), 0, "Type"},
    {NULL}
};

PyTypeObject Axis::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "Axis",
    .tp_basicsize = sizeof(Axis::Object),
    .tp_itemsize = 0,
    .tp_dealloc = Axis::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Axis object",
    .tp_methods = nullptr,
    .tp_members = Axis::Members,
    .tp_init = nullptr,
    .tp_new = Axis::_new
};

void Axis::_dealloc(PyObject *self) {
  auto self_ = (Object *)self;
  Py_XDECREF(self_->size);
  Py_XDECREF(self_->name);
  Py_XDECREF(self_->type);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *Axis::_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->id = -1;
    self->size = Py_None;
    self->name = Py_None;
    self->type = Py_None;
  }
  return (PyObject *)self;
}

int Axis::_init(PyObject *self_pyobject, int id, const ascir::SizeExpr &size, const char *name,
                ascir::Axis::Type type) {
  auto size_object = SizeExpr::_new(&SizeExpr::Type, nullptr, nullptr);
  if (size_object == nullptr) {
    return -1;
  }

  SizeExpr::_init(size_object, false, size.nums, size.dens);
  Py_IncRef(size_object);

  auto self = (Object *)self_pyobject;
  self->id = id;
  self->size = size_object;
  self->name = PyUnicode_FromString(name);
  self->type = PyUnicode_FromString(ascir::Axis::TypeStr(type));
  return 0;
}
} // namespace pyascir for Axis

/** Operator */
namespace pyascir {
PyMemberDef Operator::Members[] = {
    {"name", T_OBJECT_EX, offsetof(Operator::Object, name)},
    {"type", T_OBJECT_EX, offsetof(Operator::Object, type)},
    {nullptr}};

PyMethodDef Operator::Methods[] = {
    {nullptr}};

PyTypeObject Operator::Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "Operator",
    .tp_basicsize = sizeof(Operator::Object),
    .tp_itemsize = 0,
    .tp_dealloc = Operator::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Operator.",
    .tp_methods = Operator::Methods,
    .tp_members = Operator::Members,
    .tp_init = Operator::_init,
    .tp_new = Operator::_new
};


void Operator::_dealloc(PyObject* self_pyobject) {
  auto self = (Object*)self_pyobject;

  Py_XDECREF(self->name);
  Py_XDECREF(self->type);

  if (self->op != nullptr) {
    // May the derived operator class will change and delete the op
    delete self->op;
  }

  Py_TYPE(self)->tp_free(self_pyobject);
}

PyObject *Operator::_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)type->tp_alloc(type, 0);
  if (self == nullptr) {
    return nullptr;
  }

  self->name = nullptr;
  self->type = nullptr;
  self->op = nullptr;

  return (PyObject*)self;
}

int Operator::_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  auto self = (Object*)self_pyobject;

  char* name;
  char* type;
  if (!PyArg_ParseTuple(args, "ss", &name, &type)) {
    return -1;
  }

  self->name = PyUnicode_FromString(name);
  self->type = PyUnicode_FromString(type);
  self->op = new ge::Operator(name, type);

  return 0;
}
} // namespace pyascir for Operator

/** HintGraph */
namespace pyascir {
void HintGraph::_dealloc(PyObject *self_pyobject) {
  auto self = (Object *)self_pyobject;

  Py_XDECREF(self->name);
  Py_XDECREF(self);

  delete self->graph;

  Py_TYPE(self_pyobject)->tp_free(self_pyobject);
}

PyObject *HintGraph::_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)type->tp_alloc(type, 0);
  if (self == nullptr) {
    return nullptr;
  }

  self->name = nullptr;
  self->graph = nullptr;

  return (PyObject *)self;
}

int HintGraph::_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
  auto self = (Object *)self_pyobject;

  char *name;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return -1;
  }

  self->name = PyUnicode_FromString(name);
  self->graph = new ascir::Graph(name);

  return 0;
}

PyObject *HintGraph::_create_size(PyObject *self_pyobject, PyObject *args) {
  char *size_var_name = nullptr;
  if (!PyArg_ParseTuple(args, "s", &size_var_name)) {
    return nullptr;
  }

  auto self = (Object *)self_pyobject;
  auto new_size = self->graph->CreateSizeVar(size_var_name);
  if (new_size.id == ascir::ID_NONE) {
    return nullptr;
  }

  auto new_size_pyobject = SizeVar::_new(&SizeVar::Type, nullptr, nullptr);
  if (new_size_pyobject == nullptr) {
    return nullptr;
  }

  SizeVar::_init(new_size_pyobject, new_size.id, new_size.value, new_size.name.c_str(), ascir::SizeVar::TypeStr(new_size.type));

  Py_INCREF(new_size_pyobject);
  return new_size_pyobject;
}

PyObject *HintGraph::_create_axis(PyObject *self_pyobject, PyObject *args) {
  char* name;
  PyObject *size;
  if (!PyArg_ParseTuple(args, "sO!", &name, &SizeExpr::Type, &size)) {
    return nullptr;
  }

  auto self = (Object *)self_pyobject;
  auto new_axis = self->graph->CreateAxis(name, SizeExpr::AsSizeExpr(size));

  auto axis_object = Axis::_new(&Axis::Type, nullptr, nullptr);
  if (axis_object == nullptr) {
    return nullptr;
  }
  Axis::_init(axis_object, new_axis.id, new_axis.size, new_axis.name.c_str(), new_axis.type);
  Py_INCREF(axis_object);
  return axis_object;
}

PyObject *HintGraph::_set_inputs(PyObject *self_pyobject, PyObject *args) {
  PyObject *inputs;
  if (!PyArg_ParseTuple(args, "O", &inputs)) {
    return nullptr;
  }

  std::vector<ge::Operator> input_ops;
  for (int i = 0; i < PyList_Size(inputs); i++) {
    auto op = PyList_GetItem(inputs, i);
    if (!PyObject_IsInstance(op, (PyObject*)&Operator::Type)) {
      return PyErr_Format(PyExc_ValueError, "Input %d is not an Operator", i);
    }

    input_ops.push_back(*((Operator::Object *)op)->op);
  }

  auto self = (Object *)self_pyobject;
  self->graph->SetInputs(input_ops);

  Py_RETURN_NONE;
}

PyObject *HintGraph::_set_outputs(PyObject *self_pyobject, PyObject *args) {
  PyObject *outputs;
  if (!PyArg_ParseTuple(args, "O", &outputs)) {
    return nullptr;
  }

  std::vector<ge::Operator> output_ops;
  for (int i = 0; i < PyList_Size(outputs); i++) {
    auto op = PyList_GetItem(outputs, i);
    if (!PyObject_IsInstance(op, (PyObject*)&Operator::Type)) {
      return PyErr_Format(PyExc_ValueError, "Output %d is not an Operator", i);
    }

    output_ops.push_back(*((Operator::Object *)op)->op);
  }

  auto self = (Object *)self_pyobject;
  self->graph->SetOutputs(output_ops);

  Py_RETURN_NONE;
}

PyObject *HintGraph::FromGraph(ascir::Graph *graph) {
  auto graph_object = (HintGraph::Object*)HintGraph::_new(&HintGraph::Type, nullptr, nullptr);
  if (graph_object == nullptr) {
    return nullptr;
  }

  graph_object->name = PyUnicode_FromString(graph->GetName().c_str());
  graph_object->graph = new ascir::HintGraph(graph->GetName().c_str());
  if (graph_object->graph == nullptr) {
      return nullptr;
  }

  graph_object->graph->CopyFrom(*graph);
  Py_INCREF(graph_object);
  return (PyObject *)graph_object;
}

PyMemberDef HintGraph::Members[] = {
    {"name", T_OBJECT_EX, offsetof(HintGraph::Object, name), 0, NULL},
    {NULL}
};

PyMethodDef HintGraph::Methods[] = {
    {"create_size", (PyCFunction)HintGraph::_create_size, METH_VARARGS, "Create a size variable"},
    {"create_axis", (PyCFunction)HintGraph::_create_axis, METH_VARARGS, "Create an axis"},
    {"set_inputs", (PyCFunction)HintGraph::_set_inputs, METH_VARARGS, "Set graph inputs"},
    {"set_outputs", (PyCFunction)HintGraph::_set_outputs, METH_VARARGS, "Set graph outputs"},
    {NULL}
};

PyTypeObject HintGraph::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "HintGraph",
    .tp_basicsize = sizeof(HintGraph::Object),
    .tp_itemsize = 0,
    .tp_dealloc = HintGraph::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "HintGraph object",
    .tp_methods = HintGraph::Methods,
    .tp_members = HintGraph::Members,
    .tp_init = HintGraph::_init,
    .tp_new = HintGraph::_new,
    };
} // namespace pyascir
