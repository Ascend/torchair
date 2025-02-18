__all__ = ['custom_op']

from typing import Dict, Union, List, Optional, Tuple
from torchair.ge.attr import _Attr
from torchair.ge._ge_graph import Tensor
from torchair._ge_concrete_graph.compat_ir import ge_op


def custom_op(op_type: str, *, inputs: Optional[Dict[str, Optional[Union['Tensor', List['Tensor']]]]],
              outputs: Optional[List[Union[str, Tuple[str, int]]]], attrs: Optional[Dict[str, '_Attr']] = None,
              node_name: Optional[str] = None):
    return ge_op(op_type=op_type, inputs=inputs, outputs=outputs, attrs=attrs, node_name=node_name)


