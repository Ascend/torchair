#include "pyascir.h"

#include <Python.h>
#include <structmember.h>

#include "graph/operator.h"

#include "ascir.h"
#include "ascir_utils.h"
#include "ascir_ops.h"

#include "pyascir_types.h"

/** AttrHolder */
namespace pyascir {
class AttrHolder {
 public:
  struct Object {
    PyObject_HEAD
    ge::AttrHolder* holder;
  };

  static void _dealloc(PyObject *self);
  static int _init(PyObject *self, ge::AttrHolder* holder);
  static PyTypeObject CreateType(const char *name, PyGetSetDef* getSets, const char* doc);
  static PyObject* FromOp(PyTypeObject* type, ge::Operator* op);
};

void AttrHolder::_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

int AttrHolder::_init(PyObject *self_pyobject, ge::AttrHolder *holder) {
  auto self = (Object *)self_pyobject;
  self->holder = holder;
  return 0;
}

PyObject *AttrHolder::FromOp(PyTypeObject* type, ge::Operator *op) {
  if (op == nullptr) {
    return nullptr;
  }

  auto self = type->tp_alloc(type, 0);
  if (self == nullptr) {
    return nullptr;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  AttrHolder::_init(self, ge::AttrUtils::AttrHolderAdapter(op_desc).get());
  Py_IncRef(self);
  return self;
}

PyTypeObject AttrHolder::CreateType(const char *name, PyGetSetDef* getSets, const char* doc) {
  PyTypeObject type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = name,
    .tp_basicsize = sizeof(AttrHolder::Object),
    .tp_itemsize = 0,
    .tp_dealloc = AttrHolder::_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = doc,
    .tp_methods = nullptr,
    .tp_members = nullptr,
    .tp_getset = getSets,
  };

  return type;
};
} // namespace pyascir for AttrHolder

/** AttrFieldGetSet */
namespace pyascir {

/** GetSetters for attribute from AttrHolder */
template <const char* ATTR_NAME, typename T>
class AttrFieldGetSet {
 public:
  static PyObject *get(PyObject *self, void *closure);
  static int set(PyObject *self, PyObject *value, void *closure);
};

template <const char* ATTR_NAME, typename T>
PyGetSetDef AttrHolderGetSetDef(const char *name, const char *doc) {
    using t = AttrFieldGetSet<ATTR_NAME, T>;
    return {name, (getter)t::get, (setter)t::set, doc, nullptr};
}

template <const char *ATTR_NAME>
class AttrFieldGetSet<ATTR_NAME, int64_t> {
 public:
  static PyObject *get(PyObject *self, void *closure) {
    auto attr_struct = (AttrHolder::Object*)self;

    int64_t value = 0;
    ge::AttrUtils::GetInt(attr_struct->holder, ATTR_NAME, value);
    return PyLong_FromLong(value);
  }

  static int set(PyObject *self, PyObject *value, void *closure) {
    int64_t val = PyLong_AsLong(value);
    auto attr_struct = (AttrHolder::Object *)self;
    ge::AttrUtils::SetInt(attr_struct->holder, ATTR_NAME, val);
    return 0;
  }
};

template <const char *ATTR_NAME>
class AttrFieldGetSet<ATTR_NAME, std::vector<ascir::SizeExpr>> {
 public:
  static PyObject *get(PyObject *self, void *closure) {
    auto attr_struct = (AttrHolder::Object*)self;

    ascir::AttrField<ge::AttrHolder*, ATTR_NAME, std::vector<ascir::SizeExpr>> field{attr_struct->holder};
    auto value = field();

    PyObject* size_list = PyList_New(value.size());
    for (auto &size : value) {
        PyList_Append(size_list, SizeExpr::FromSizeExpr(size));
    }

    return size_list;
  }

  static int set(PyObject *self, PyObject *value, void *closure) {
    if (!PyList_Check(value)) {
        return -1;
    }

    std::vector<ascir::SizeExpr> size_list;
    auto attr_struct = (AttrHolder::Object *)self;
    for (int i = 0; i < PyList_Size(value); i++) {
      auto size_expr = PyList_GetItem(value, i);
      if (!PyObject_IsInstance(size_expr, (PyObject*)&SizeExpr::Type)) {
        PyErr_Format(PyExc_ValueError, "size expression on %d is not SizeExpr type", i);
        return -1;
      }

      size_list.push_back(SizeExpr::AsSizeExpr(size_expr));
    }

    ascir::AttrField<ge::AttrHolder*, ATTR_NAME, std::vector<ascir::SizeExpr>> field{attr_struct->holder};
    field = size_list;
    return 0;
  }
};

template <const char* ATTR_NAME>
class AttrFieldGetSet<ATTR_NAME, std::vector<ascir::AxisId>> {
 public:
  static PyObject *get(PyObject *self, void *closure) {
    auto attr_struct = (AttrHolder::Object*)self;

    std::vector<ascir::AxisId> value;
    ge::AttrUtils::GetListInt(attr_struct->holder, ATTR_NAME, value);

    auto list = PyList_New(value.size());
    for (int i = 0; i < value.size(); ++i) {
      PyList_SetItem(list, i, PyLong_FromLong(value[i]));
    }
    return list;
  }

  static int set(PyObject *self, PyObject *value, void *closure) {
    if (!PyList_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "value must be a list");
      return -1;
    }

    std::vector<ascir::AxisId> axis_id_list;
    for (int i = 0; i < PyList_Size(value); ++i) {
      auto axis = (Axis::Object*)PyList_GetItem(value, i);
      if (!PyObject_IsInstance((PyObject*)axis, (PyObject*)&Axis::Type)) {
        PyErr_Format(PyExc_ValueError, "axis on %d is not Axis type", i);
        return -1;
      }
      axis_id_list.push_back(axis->id);
    }

    auto attr_struct = (AttrHolder::Object *)self;
    ge::AttrUtils::SetListInt(attr_struct->holder, ATTR_NAME, axis_id_list);
    return 0;
  }
};
} // namespace pyascir for AttrFieldGetSet

/** OpsOperatorAttr include sched hint */
namespace pyascir {
class OpsOperatorAttrSched {
 public:
  static PyTypeObject Type;
  static PyGetSetDef GetSetters[];
  static PyObject* FromOp(ge::Operator* op);
};

PyGetSetDef OpsOperatorAttrSched::GetSetters[] = {
    AttrHolderGetSetDef<ascir::NodeAttr::SCHED_EXEC_ORDER, int64_t>("exec_order", "Execute order."),
    AttrHolderGetSetDef<ascir::NodeAttr::SCHED_AXIS, std::vector<ascir::AxisId>>("axis", "Axis of scheduler."),
    {NULL} /* Sentinel */
};

PyTypeObject OpsOperatorAttrSched::Type = AttrHolder::CreateType(
    "OpsOperatorAttrSched", OpsOperatorAttrSched::GetSetters, "Operator schedule attributes.");

PyObject *OpsOperatorAttrSched::FromOp(ge::Operator *op) {
  return AttrHolder::FromOp(&OpsOperatorAttrSched::Type, op);
}

class OpsOperatorAttr {
 public:
  struct Object{
    PyObject_HEAD
    PyObject* sched; // OpsOperatorAttrSched
  };

  static PyMemberDef Members[];
  static PyTypeObject Type;

  static void _dealloc(PyObject *self);
  static PyObject* FromOp(ge::Operator* op);
};

PyMemberDef OpsOperatorAttr::Members[] = {
  {"sched", T_OBJECT_EX, offsetof(Object, sched), 0, nullptr},
  {nullptr}
};

PyTypeObject OpsOperatorAttr::Type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
 .tp_name = "OpsOperatorAttr",
 .tp_basicsize = sizeof(OpsOperatorAttr::Object),
 .tp_itemsize = 0,
 .tp_dealloc = OpsOperatorAttr::_dealloc,
 .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
 .tp_doc = "Operator attributes.",
 .tp_methods = nullptr,
 .tp_members = OpsOperatorAttr::Members,
 .tp_init = nullptr,
 .tp_new = nullptr
};

void OpsOperatorAttr::_dealloc(PyObject *self) {
  auto self_ = (Object *)self;
  Py_XDECREF(self_->sched);
  Py_TYPE(self)->tp_free(self);
}

PyObject *OpsOperatorAttr::FromOp(ge::Operator *op) {
  if (op == nullptr) {
    return nullptr;
  }

  auto attr = (OpsOperatorAttr::Object*)OpsOperatorAttr::Type.tp_alloc(&OpsOperatorAttr::Type, 0);
  if (attr == nullptr) {
    return nullptr;
  }

  attr->sched = OpsOperatorAttrSched::FromOp(op);
  Py_IncRef((PyObject*)attr);
  return (PyObject *)attr;
}
} // namespace pyascir for OpsOperatorAttr

/** OpsOperator and input/output */
namespace pyascir {
class OpsOperatorInput {
 public:
  static int _setter(PyObject *self, PyObject *value, void *closure);
};

class OpsOperatorOutput {
 public:
  struct Object {
    AttrHolder::Object attr_holder;
    int index;
    ge::Operator* op;
  };

  static PyGetSetDef GetSetters[];
  static PyTypeObject Type;
  static void _dealloc(PyObject *self);
  static PyObject* _get_dtype(PyObject *self, void *closure);
  static void _set_dtype(PyObject *self, PyObject *value, void *closure);
  static PyObject* FromOp(int index, ge::Operator* op);
  static PyMemberDef CreateMember(const char* name, Py_ssize_t offset);
};

PyGetSetDef OpsOperatorOutput::GetSetters[] = {
  AttrHolderGetSetDef<ascir::TensorAttr::AXIS, std::vector<ascir::AxisId>>("axis", "Axis"),
  AttrHolderGetSetDef<ascir::TensorAttr::REPEATS, std::vector<ascir::SizeExpr>>("size", "Size along each axis"),
  AttrHolderGetSetDef<ascir::TensorAttr::STRIDES, std::vector<ascir::SizeExpr>>("strides", "Stride along each axis"),
  {"dtype", (getter)OpsOperatorOutput::_get_dtype, (setter)OpsOperatorOutput::_set_dtype, "Data type"},
  {nullptr} /* Sentinel */
};

PyTypeObject OpsOperatorOutput::Type = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  .tp_name = "OpsOperatorOutput",
  .tp_basicsize = sizeof(OpsOperatorOutput::Object),
  .tp_itemsize = 0,
  .tp_dealloc = OpsOperatorOutput::_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_getset = OpsOperatorOutput::GetSetters,
};

void OpsOperatorOutput::_dealloc(PyObject *self) {
  Py_TYPE(self)->tp_free(self);
}

void OpsOperatorOutput::_set_dtype(PyObject *self, PyObject *value, void *closure) {
  auto dtype = PyLong_AsLong(value);
  auto self_ = (OpsOperatorOutput::Object *)self;

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*self_->op);
  op_desc->MutableOutputDesc(self_->index)->SetDataType(static_cast<ge::DataType>(dtype));
}

PyObject* OpsOperatorOutput::_get_dtype(PyObject *self, void *closure) {
  auto self_ = (OpsOperatorOutput::Object *)self;
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*self_->op);
  auto dtype = op_desc->GetOutputDesc(self_->index).GetDataType();

  return PyLong_FromLong(dtype);
}

PyObject *OpsOperatorOutput::FromOp(int index, ge::Operator *op) {
  auto self = (Object*)Type.tp_alloc(&Type, 0);
  if (self == nullptr) {
    return nullptr;
  }

  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*op);
  auto output_desc = op_desc->MutableOutputDesc(index);
  AttrHolder::_init((PyObject*)self, ge::AttrUtils::AttrHolderAdapter(*output_desc).get());

  self->op = op;
  self->index = index;
  Py_IncRef((PyObject*)self);
  return (PyObject*)self;
}

PyMemberDef OpsOperatorOutput::CreateMember(const char *name, Py_ssize_t offset) {
  return {name, T_OBJECT_EX, offset, 0, nullptr};
}

template <typename OpType>
class OpsOperator {
public:
  struct Object {
    Operator::Object op_base;
    OpType* op;
    int input_output_num;

    PyObject *attr; // OpsOperatorAttr
    PyObject* input_outputs[0];
  };

  static void OpsOperator_dealloc(PyObject *self_pyobject) {
    auto self = (Object *)self_pyobject;
    delete self->op;
    self->op_base.op = nullptr;
    Operator::_dealloc((PyObject *)self);
  }

  static PyObject *OpsOperator_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    auto self = (Object *)Operator::_new(type, args, kwargs);
    if (self == nullptr) {
      return nullptr;
    }

    self->attr = Py_None;
    self->input_output_num = 0;
    self->op_base.op = nullptr;

    return (PyObject *)self;
  }

  static int OpsOperator_init(PyObject *self_pyobject, PyObject *args, PyObject *kwargs) {
    char* name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
      return -1;
    }

    auto op = new OpType(name);
    if (op == nullptr) {
        return -1;
    }

    auto attr = OpsOperatorAttr::FromOp(op);
    if (attr == nullptr) {
        delete op;
        return -1;
    }

    auto self = (Object *)self_pyobject;
    self->op_base.name = PyUnicode_FromString(name);
    self->op_base.type = PyUnicode_FromString(OpType::Type);
    self->op_base.op = op;
    self->op = op;
    self->attr = attr;

    for (int i = 0; i < op->GetOutputsSize(); ++i) {
      auto output = OpsOperatorOutput::FromOp(i, op);
      if (output == nullptr) {
          return -1;
      }
      self->input_outputs[self->input_output_num++] = output;
    }

    return 0;
  }

  static PyTypeObject CreateTypeObject() {
    static PyMethodDef methods[] = {{nullptr}};

    auto members = new vector<PyMemberDef>{
        {"attr", T_OBJECT_EX, offsetof(OpsOperator<OpType>::Object, attr), 0, "Operator attributes."},
    };
    auto getsetters = new vector<PyGetSetDef>{};

    // Alloc  *never* free vector to save input/output names
    // So that can set it to members
    auto input_names = std::vector<std::string>();
    auto output_names = std::vector<std::string>();

    OpType sample_op("sample");
    auto sample_op_desc = ge::OpDescUtils::GetOpDescFromOperator(sample_op);
    for (int i = 0; i < sample_op_desc->GetOutputsSize(); i++) {
        output_names.push_back(sample_op_desc->GetOutputNameByIndex(i));
    }
    for (int i = 0; i < sample_op_desc->GetInputsSize(); i++) {
        input_names.push_back(sample_op_desc->GetInputNameByIndex(i));
    }

    int input_output_num = 0;
    for (auto &name : output_names) {
        members->push_back(OpsOperatorOutput::CreateMember(
            name.c_str(),
            offsetof(Object, input_outputs) + input_output_num * sizeof(PyObject *)));
        input_output_num++;
    }
    members->push_back({nullptr});

    for (int i = 0; i < input_names.size(); ++i) {
        getsetters->push_back(
            {input_names[i].c_str(), nullptr, (setter)OpsOperatorInput::_setter, "", PyLong_FromLong(i)});
    }
    getsetters->push_back({nullptr});

    PyTypeObject type_object = {PyVarObject_HEAD_INIT(nullptr, 0)};
    type_object.tp_name = OpType::Type;
    type_object.tp_doc = OpType::Type;
    type_object.tp_base = &Operator::Type;
    type_object.tp_basicsize = sizeof(Object) + input_output_num * sizeof(PyObject *);
    type_object.tp_itemsize = 0;
    type_object.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type_object.tp_new = OpsOperator_new;
    type_object.tp_init = OpsOperator_init;
    type_object.tp_dealloc = OpsOperator_dealloc;
    type_object.tp_members = members->data();
    type_object.tp_methods = methods;
    type_object.tp_getset = getsetters->data();

    return type_object;
  }
};

int OpsOperatorInput::_setter(PyObject *self, PyObject *value, void *closure) {
  auto self_ = (OpsOperator<ge::Operator>::Object*)self;
  int index = PyLong_AsLong((PyObject*)closure);
  if (PyObject_IsInstance(value, (PyObject *)&Operator::Type)) {
    auto op = (Operator::Object*)value;
    self_->op->SetInput(index, *op->op, 0);
    return 0;
  } else if (PyObject_IsInstance(value, (PyObject *)&OpsOperatorOutput::Type)) {
    auto output = (OpsOperatorOutput::Object *)value;
    self_->op->SetInput(index, *output->op, output->index);
    return 0;
  } else {
    return -1;
  }

  return 0;
}
} // namespace pyascir for OpsOperator

static PyModuleDef GraphModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "graph",
    .m_doc = "Graph module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_graph(void) {
    PyObject *m;
    m = PyModule_Create(&GraphModule);
    if (m == NULL) {
        return NULL;
    }
    return m;
}

static PyObject *UtilsDebugStr(PyObject *self_pyobject, PyObject *args) {
    PyObject *graph_object;
    if (!PyArg_ParseTuple(args, "O!", &pyascir::HintGraph::Type, &graph_object)) {
        return PyErr_Format(PyExc_TypeError, "Argument must be a HintGraph object");
    }

    auto graph = (pyascir::HintGraph::Object *)graph_object;
    auto debug_str = ascir::utils::DebugStr(*graph->graph);
    return PyUnicode_FromString(debug_str.c_str());
}

static PyObject *UtilsDumpGraph(PyObject *self_pyobject, PyObject *args) {
  Py_RETURN_NONE;
}

static PyMethodDef UtilsMethods[] = {
    {"debug_str", (PyCFunction)UtilsDebugStr, METH_VARARGS, "Get graph debug string"},
    {"dump", (PyCFunction)UtilsDumpGraph, METH_VARARGS, "Dump graph"},
    {NULL}
};

static PyModuleDef UtilsModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "utils",
    .m_doc = "Utils module",
    .m_size = -1,
    .m_methods = UtilsMethods,
};

static PyModuleDef OpsModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ops",
    .m_doc = "Operators that ASCIR supports",
    .m_size = -1,
};

static PyModuleDef AscirModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ascir",
    .m_doc = "AscendC IR",
    .m_size = -1,
};

static PyModuleDef DtypesModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "dtypes",
    .m_doc = "Data types",
    .m_size = -1,
};

static int OpsModule_init(PyObject* ascir_module) {
  auto module_types = std::vector{
      std::pair{"SizeVar", &pyascir::SizeVar::Type},
      std::pair{"SizeExpr", &pyascir::SizeExpr::Type},
      std::pair{"Axis", &pyascir::Axis::Type},

      std::pair{"OperatorOutput", &pyascir::OpsOperatorOutput::Type},
      std::pair{"OpsOperatorAttr", &pyascir::OpsOperatorAttr::Type},
      std::pair{"OpsOperatorAttrSched", &pyascir::OpsOperatorAttrSched::Type},
      std::pair{"Operator", &pyascir::Operator::Type},

      std::pair{"HintGraph", &pyascir::HintGraph::Type},
  };
  for (auto [name, type] : module_types) {
    if (PyType_Ready(type) < 0) {
      return -1;
    }
    Py_INCREF(type);
    PyModule_AddObject(ascir_module, name, (PyObject*)type);
  }

  return 0;
}

PyMODINIT_FUNC PyInit_ascir(void) {
  PyObject *ascir_module = PyModule_Create(&AscirModule);
  if (ascir_module == NULL) {
    return nullptr;
  }

  PyObject *dtypes_module = PyModule_Create(&DtypesModule);
  if (dtypes_module == NULL) {
    return nullptr;
  }

  PyModule_AddObject(dtypes_module, "float32", PyLong_FromLong(ge::DT_FLOAT));
  PyModule_AddObject(dtypes_module, "float16", PyLong_FromLong(ge::DT_FLOAT16));
  PyModule_AddObject(ascir_module, "dtypes", dtypes_module);

  if (OpsModule_init(ascir_module) < 0) {
    return nullptr;
  }

  static auto utils_module = PyModule_Create(&UtilsModule);
  if (utils_module == nullptr) {
    return nullptr;
  }

  PyModule_AddObject(ascir_module, "utils", utils_module);

  static auto ops_module = PyModule_Create(&OpsModule);
  if (ops_module == nullptr) {
    return nullptr;
  }

  static PyTypeObject ops_operators[] = {
    pyascir::OpsOperator<ascir::ops::Data>::CreateTypeObject(),
    pyascir::OpsOperator<ascir::ops::Load>::CreateTypeObject(),
    pyascir::OpsOperator<ascir::ops::Abs>::CreateTypeObject(),
    pyascir::OpsOperator<ascir::ops::Store>::CreateTypeObject(),
  };
  for (auto &type : ops_operators) {
    if (PyType_Ready(&type) < 0) {
      return nullptr;
    }
    Py_INCREF(&type);
    PyModule_AddObject(ops_module, type.tp_name, (PyObject*)&type);
  }

  PyModule_AddObject(ascir_module, "ops", ops_module);
  return ascir_module;
}
