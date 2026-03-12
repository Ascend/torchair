from ._scope import _npu_scope_enter, _npu_scope_exit


class _Scope:
    def __init__(self, attrs):
        self.attrs = attrs

    def __enter__(self):
        return _npu_scope_enter(self.attrs)

    def __exit__(self, *args):
        return _npu_scope_exit()


def limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    return _Scope([
        ("_op_aicore_num", str(op_aicore_num)),
        ("_op_vectorcore_num", str(op_vectorcore_num))
    ])
