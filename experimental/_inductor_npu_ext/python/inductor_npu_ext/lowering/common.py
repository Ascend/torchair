import os
import atexit
from typing import Dict, Tuple, List, Union

import torch

import torch.utils._pytree as pytree
from torch._ops import OpOverload, OpOverloadPacket

from inductor_npu_ext.common import logger
from inductor_npu_ext.common.utils import get_node_meta


class LowerSummary:
    def __init__(self):
        self.enabled = os.getenv("TORCH_COMPILE_DEBUG", "0") == "1"
        self.lowered_ops = dict()
        self.fallback_ops = dict()

    def _node_key(self, node: torch.fx.Node):
        in_nodes = [n for n in pytree.arg_tree_leaves(*node.args, **node.kwargs) if isinstance(n, torch.fx.Node)]
        input_metas = ','.join([f'Tensor({m.dtype}, {m.shape}, {m.device})' if isinstance(
            m, torch.Tensor) else str(m) for m in get_node_meta(in_nodes)])
        out_metas = ','.join([f'Tensor({m.dtype}, {m.shape}, {m.device})' if isinstance(
            m, torch.Tensor) else str(m) for m in get_node_meta([node])])
        key = f"{node.target}({input_metas}) -> ({out_metas})"
        return key

    def lowered(self, node: torch.fx.Node):
        if not self.enabled:
            return
        key = self._node_key(node)
        self.lowered_ops[key] = self.lowered_ops.get(key, 0) + 1

    def fallback(self, node: torch.fx.Node, reason):
        if not self.enabled:
            return
        key = self._node_key(node) + f"  # reason: {reason}"
        self.fallback_ops[key] = self.fallback_ops.get(key, 0) + 1

    def save(self):
        if not self.enabled:
            return

        for key, count in self.lowered_ops.items():
            logger.info(f"Lowered {count}x {key}")

        for key, count in self.fallback_ops.items():
            logger.info(f"Fallback {count}x {key}")


def exclude(*args):
    # no complex dtypes for now
    # no torch.int64 or torch.uint16/32/64 for npu
    # no torch.float64 for now
    all_dtypes = {torch.int8, torch.int16, torch.int32,
                  torch.uint8, torch.float16, torch.float32,
                  torch.bfloat16, torch.bool}
    return tuple(dtype for dtype in all_dtypes if dtype not in args)


def float_dtypes():
    return (torch.float16, torch.float32, torch.bfloat16)


def byte_dtypes():
    return (torch.uint8, torch.bool)


class _LoweringGuard:
    _datas: Dict[OpOverload, Tuple[Tuple[torch.dtype], Tuple[torch.dtype]]] = {}

    @classmethod
    def has(cls, op: OpOverload):
        return op in cls._datas.keys()

    @classmethod
    def dtypes_support(cls, op: OpOverload):
        return cls._datas.get(op)

    @classmethod
    def support(cls, ops: Union[OpOverload, OpOverloadPacket, Tuple[OpOverload]],
                support_in_dtypes: Tuple[torch.dtype],
                support_out_dtypes: Tuple[torch.dtype] = None):
        support_out_dtypes = support_out_dtypes if support_out_dtypes is not None else support_in_dtypes
        if isinstance(ops, OpOverloadPacket):
            ops = [getattr(ops, overload) for overload in ops.overloads()]
        elif isinstance(ops, OpOverload):
            ops = [ops]
        for op in ops:
            cls._datas[op] = (support_in_dtypes, support_out_dtypes)


_summary = LowerSummary()
atexit.register(lambda: _summary.save())
