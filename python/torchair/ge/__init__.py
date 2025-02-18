__all__ = ['custom_op', 'Const', 'Cast', 'DataType', 'Format', 'Tensor', 'TensorSpec']

from torchair.ge.ge_custom import custom_op
from torchair.ge._ge_graph import Const as _Const
from torchair.ge._ge_graph import Cast as _Cast
from torchair.ge._ge_graph import TensorBase as Tensor
from torchair.ge._ge_graph import TensorSpecBase as TensorSpec
from torchair.ge._ge_graph import DataType
from torchair.ge._ge_graph import Format


from . import attr


def Const(v, dtype: int = None, node_name=None, readable=True):
    return _Const(v=v, dtype=dtype, node_name=node_name, readable=readable)


def Cast(x: Tensor, *, dst_type: int, dependencies=[], node_name=None):
    return _Cast(x=x, dst_type=dst_type, dependencies=dependencies, node_name=node_name)
