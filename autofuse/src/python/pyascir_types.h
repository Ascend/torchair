#ifndef __PYASCIR_TYPES_H__
#define __PYASCIR_TYPES_H__

#include <Python.h>
#include <structmember.h>

#include "ascir.h"

namespace pyascir {
class SizeVar {
 public:
  struct Object {
    PyObject_HEAD

    int id;
    int value;
    PyObject* name;
    PyObject* type;
  };

  static PyTypeObject Type;
  static PyMemberDef Members[];

  static void _dealloc(PyObject *self);
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int _init(PyObject *self, int id, int value, const char *name, const char *type);
  static int _init(PyObject *self, PyObject *args, PyObject *kwargs);
};

class SizeExpr {
 public:
  struct Object {
    PyObject_HEAD

    PyObject* nums; // list of nums
    PyObject* dens; // list of dens
  };

  static PyTypeObject Type;
  static PyMemberDef Members[];

  static void _dealloc(PyObject *self);
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int _init(PyObject *self, PyObject *args, PyObject *kwargs);
  static int _init(PyObject *self, const std::vector<ascir::SizeVarId> &nums,
                           const std::vector<ascir::SizeVarId> &dens);
  static PyObject* FromSizeExpr(const ascir::SizeExpr &expr);
  static ascir::SizeExpr AsSizeExpr(PyObject* obj);
};

class Axis {
 public:
  struct Object {
    PyObject_HEAD

    int id;
    PyObject *size;
    PyObject *name;
    PyObject *type;
  };

  static PyTypeObject Type;
  static PyMemberDef Members[];

  static void _dealloc(PyObject *self);
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int _init(PyObject *self, int id, const ascir::SizeExpr& size, const char *name, ascir::Axis::Type type);
};

class Operator {
 public:
  struct Object {
    PyObject_HEAD

    PyObject *name;
    PyObject *type;
    ge::Operator* op;
  };

  static PyTypeObject Type;
  static PyMemberDef Members[];
  static PyMethodDef Methods[];

  static void _dealloc(PyObject *self);
  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwargs);
  static int _init(PyObject *self, PyObject* args, PyObject* kwargs);
};

class HintGraph {
 public:
  struct Object {
    PyObject_HEAD

    PyObject *name;
    ascir::Graph* graph;
  };

  static PyTypeObject Type;
  static PyMemberDef Members[];
  static PyMethodDef Methods[];

  static void _dealloc(PyObject* self_pyobject);
  static PyObject* _new(PyTypeObject* type, PyObject* args, PyObject* kwargs);
  static int _init(PyObject* self_pyobject, PyObject* args, PyObject* kwargs);
  static PyObject* _create_size(PyObject* self_pyobject, PyObject* args);
  static PyObject* _create_axis(PyObject* self_pyobject, PyObject* args);
  static PyObject* _set_inputs(PyObject* self_pyobject, PyObject* args);
  static PyObject* _set_outputs(PyObject* self_pyobject, PyObject* args);

  static PyObject* FromGraph(ascir::Graph* graph);
};
}

#endif
