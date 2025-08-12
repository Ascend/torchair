__all__ = ['custom_op', 'Const', 'Cast', 'DataType', 'Format', 'Tensor', 'TensorSpec',
           'Clone']

from torchair.ge.ge_custom import custom_op
from torchair.ge._ge_graph import Const as _Const
from torchair.ge._ge_graph import Cast as _Cast
from torchair.ge._ge_graph import TensorBase as Tensor
from torchair.ge._ge_graph import TensorSpecBase as TensorSpec
from torchair.ge._ge_graph import DataType
from torchair.ge._ge_graph import Format
from torchair._ge_concrete_graph.ge_apis import TensorMove

from . import attr


def Const(v, dtype: int = None, node_name=None, readable=True):
    return _Const(v=v, dtype=dtype, node_name=node_name, readable=readable)


def Cast(x: Tensor, *, dst_type: int, dependencies=[], node_name=None):
    return _Cast(x=x, dst_type=dst_type, dependencies=dependencies, node_name=node_name)


def Clone(x: Tensor, *, dependencies=[], node_name=None):
    def check_input_is_tensor(target):
        if not isinstance(target, Tensor):
            raise AssertionError(f"input '{target}' need to be Tensor, please check your input")
    check_input_is_tensor(x)
    if dependencies is not None:
        for dependency in dependencies:
            check_input_is_tensor(dependency)
    return TensorMove(x=x, dependencies=dependencies, node_name=node_name)