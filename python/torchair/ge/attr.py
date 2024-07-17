_ListTypeMap = {
    "list.s": 0,
    "list.i": 2,
    "list.f": 3,
    "list.b": 4,
    "list.dt": 10
}


class _Attr:
    def __init__(self, value, field, setter, getter, *, cls):
        self.value = value
        self.field = field
        self.cls = cls
        self.setter = setter
        self.getter = getter
        self.list_type = None if self.field not in _ListTypeMap else _ListTypeMap[self.field]

    def __call__(self, v):
        return self._wrap(v)

    def __eq__(self, other):
        return self.value == other.value and self.cls == other.cls

    def __repr__(self):
        return f"ge.attr.{self.cls}({repr(self.value)})"

    def merge_to(self, obj):
        self.setter(obj, self.value)

    def get(self, obj):
        if self.list_type is None:
            if not obj.HasField(self.field):
                return None
        elif not obj.HasField('list') or obj.list.val_type != self.list_type:
            return None
        return self._wrap(self.getter(obj))

    def _wrap(self, value):
        if isinstance(value, type(self)):
            if value.field != self.field:
                raise AssertionError(f"Type mismatch: {self.field} != {value.field}")
            return value
        value = self.value if value is None else value
        if value is None:
            raise AssertionError(f"Value cannot be None.")
        return _Attr(value, self.field, self.setter, self.getter, cls=self.cls)


def _make_attr_builder(field, setter=" = {}", getter="{}", *, list_type=None, cls):
    set_command = f'obj.{field}{setter.format("value")}'
    if list_type is not None:
        set_command += f'\nobj.list.val_type = {list_type}'  # set list type
    attr_obj = f'obj.{field}'
    get_command = f'{getter.format(attr_obj)}'

    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import compat_as_bytes
    from torchair.ge._ge_graph import compat_as_bytes_list

    used_function = {
        'trans_to_list_list_int': trans_to_list_list_int,
        'trans_to_list_list_float': trans_to_list_list_float,
        'compat_as_bytes': compat_as_bytes,
        'compat_as_bytes_list': compat_as_bytes_list
    }

    def setter(obj, value):
        exec(set_command, used_function, locals())

    def getter(obj):
        return eval(get_command, used_function, locals())

    setter = setter

    return _Attr(None, field, setter, getter, cls=cls)


def _make_list_attr_builder(field, setter=".extend({})", getter="{}", *, cls):
    list_type = None if field not in _ListTypeMap else _ListTypeMap[field]
    return _make_attr_builder(field, setter, getter, list_type=list_type, cls=cls)


Int = _make_attr_builder("i", cls="Int")
Float = _make_attr_builder("f", cls="Float")
Bool = _make_attr_builder("b", cls="Bool")
DataType = _make_attr_builder("dt", cls="DataType")
ListInt = _make_list_attr_builder("list.i", cls="ListInt")
ListFloat = _make_list_attr_builder("list.f", cls="ListFloat")
ListBool = _make_list_attr_builder("list.b", cls="ListBool")
ListDataType = _make_list_attr_builder("list.dt", cls="ListDataType")
Str = _make_attr_builder("s", " = compat_as_bytes({})", "{}.decode()", cls="Str")
ListStr = _make_list_attr_builder("list.s", ".extend(compat_as_bytes_list({}))", "[s.decode() for s in {}]",
                                  cls="ListStr")
ListListInt = _make_attr_builder("list_list_int", ".CopyFrom(trans_to_list_list_int({}))",
                                 "[[x for x in v.list_i] for v in {}.list_list_i]", cls="ListListInt")
ListListFloat = _make_attr_builder("list_list_float", ".CopyFrom(trans_to_list_list_float({}))",
                                   "[[x for x in v.list_f] for v in {}.list_list_f]", cls="ListListFloat")
