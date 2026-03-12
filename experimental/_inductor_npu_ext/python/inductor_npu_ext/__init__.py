import atexit
import functools
import os
import sys
import itertools
from typing import Dict, Tuple, List, Union

import torch

import torch.utils._pytree as pytree
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor import graph as torch__inductor_graph
from torch._ops import OpOverload, OpOverloadPacket

import torch_npu
from inductor_npu_ext.npu import NPUScheduling, NpuWrapperCodeGen
from inductor_npu_ext.common import logger


if sys.modules.setdefault('torch_npu._inductor', None):
    raise ImportError("torch_npu._inductor already registered as codegen backend")
register_backend_for_device("npu", NPUScheduling, NpuWrapperCodeGen)

prims = torch.ops.prims
aten = torch.ops.aten


def _finetune_inductor_config():
    """
    Fine-tunes TorchInductor and TorchDynamo configuration for NPU extension.

    This function adjusts several internal TorchInductor and TorchDynamo settings to optimize
    compilation and execution for NPU devices. It modifies recompilation limits, float specialization,
    reduction unrolling, size assertions, and device properties. It also restricts operator decompositions
    to a whitelist and sets a custom pre-gradient pass if not already set.
    """
    from torch._inductor.codegen import cpu_device_op_overrides
    from torch._dynamo import config as dynamo_config
    dynamo_config.recompile_limit = 16  # default 8 is always not enough for per-layer compile
    dynamo_config.specialize_float = True  # enable float specialization until launch with scalar supported
    from torch._inductor import config as inductor_config
    inductor_config.unroll_reductions_threshold = 1  # disable unroll reductions
    inductor_config.size_asserts = False  # npu ops always return contiguous tensors which maybe different from meta outputs
    from torch._inductor.runtime.hints import DeviceProperties
    DeviceProperties.multi_processor_count = 0  # disable multi processor count
    from torch._inductor.decomposition import decompositions
    decompositions_whitelist = {
        k: v for k, v in decompositions.items() if k in {aten._to_copy.default}
    }
    decompositions.clear()
    decompositions.update(decompositions_whitelist)

    import inductor_npu_ext.lowering  # noqa: F401, make sure lowering rules for npu opps are registered

    from inductor_npu_ext.passes.auto_functionalize_legacy_ops import auto_functionalize_legacy_ops
    if inductor_config.pre_grad_custom_pass is None:
        inductor_config.pre_grad_custom_pass = auto_functionalize_legacy_ops
    else:
        if inductor_config.pre_grad_custom_pass != auto_functionalize_legacy_ops:
            logger.warning(f"`pre_grad_custom_pass` has already been registered as {inductor_config.pre_grad_custom_pass}, "
                           f"registration of `auto_functionalize_legacy_ops` will be skipped. "
                           f"Note that this may introduce extra tensormove operations. "
                           f"You can import `auto_functionalize_legacy_ops` and call it within your custom pass: "
                           f"`from inductor_npu_ext.passes.auto_functionalize_legacy_ops import auto_functionalize_legacy_ops`")


_finetune_inductor_config()


def get_node_meta(nodes: List[torch.fx.Node]):
    nodes = [nodes] if isinstance(nodes, torch.fx.Node) else nodes
    metas = []
    for n in nodes:
        if 'val' not in n.meta:
            continue
        metas.extend(pytree.tree_leaves(n.meta['val']))
    return metas


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


_summary = LowerSummary()
atexit.register(lambda: _summary.save())


def patch_fn(model, fn):
    orig_fn = getattr(model, fn)

    def decorator(f):
        @functools.wraps(orig_fn)
        def inner(*args, **kwargs):
            return f(*args, **kwargs, orig_fn=orig_fn)

        setattr(model, fn, inner)
        return inner

    return decorator


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


# basic math ops
_LoweringGuard.support(aten.add, float_dtypes())
_LoweringGuard.support(aten.exp, float_dtypes())
_LoweringGuard.support(aten.mul, float_dtypes())
_LoweringGuard.support(aten.pow, float_dtypes())
_LoweringGuard.support(aten.div, float_dtypes())
_LoweringGuard.support(aten.rsqrt, float_dtypes())
_LoweringGuard.support(aten.sqrt, float_dtypes())
_LoweringGuard.support(aten.sub, float_dtypes())
_LoweringGuard.support(aten.abs, float_dtypes())
_LoweringGuard.support(aten.floor_divide, float_dtypes())
_LoweringGuard.support(prims.convert_element_type, float_dtypes())
_LoweringGuard.support(aten.sigmoid, float_dtypes())
_LoweringGuard.support(aten.remainder, float_dtypes())
_LoweringGuard.support(aten.silu, float_dtypes())
_LoweringGuard.support(aten.neg, float_dtypes() + (torch.int32,))

# npu ops
_LoweringGuard.support(torch.ops.npu._npu_dtype_cast, float_dtypes())
_LoweringGuard.support(torch.ops.npu.npu_dtype_cast, float_dtypes())

# basic compare ops, support int32 as well
_LoweringGuard.support(aten.ge, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.le, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.gt, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.lt, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.eq, float_dtypes() + (torch.int32,))
_LoweringGuard.support(aten.ne, float_dtypes() + (torch.int32,))

# fill and create tensor ops
_LoweringGuard.support(aten.new_empty, float_dtypes())
_LoweringGuard.support(aten.detach, float_dtypes())
_LoweringGuard.support(aten.arange, float_dtypes())
_LoweringGuard.support(aten.copy_, float_dtypes())
_LoweringGuard.support(aten.copy, float_dtypes())
_LoweringGuard.support(aten.zeros_like, float_dtypes())
_LoweringGuard.support(aten.zeros, float_dtypes())
_LoweringGuard.support(aten._to_copy, float_dtypes())

# bitwise ops
_LoweringGuard.support(aten.bitwise_and, byte_dtypes())

# reduction ops
_LoweringGuard.support(aten.sum, float_dtypes())
_LoweringGuard.support(aten.mean, float_dtypes())
_LoweringGuard.support(aten.max, float_dtypes())
_LoweringGuard.support(aten.min, float_dtypes())

# view/shape ops
_LoweringGuard.support(aten.unsqueeze, float_dtypes())
_LoweringGuard.support(aten.reshape, float_dtypes())
_LoweringGuard.support(aten.squeeze, float_dtypes())
_LoweringGuard.support(aten.permute, float_dtypes())
_LoweringGuard.support(aten.select, float_dtypes())
_LoweringGuard.support(aten.slice, float_dtypes())
_LoweringGuard.support(aten._unsafe_view, float_dtypes())
_LoweringGuard.support(aten.t, float_dtypes())
_LoweringGuard.support(aten.transpose, float_dtypes())
_LoweringGuard.support(aten.expand, float_dtypes())
_LoweringGuard.support(aten.alias, float_dtypes())
_LoweringGuard.support(aten.repeat, float_dtypes())
_LoweringGuard.support(aten.sym_size, float_dtypes())


@patch_fn(torch__inductor_graph, "fallback_node_due_to_unsupported_type")
def _fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True, *, orig_fn=None):
    if orig_fn(node, allow_cpu_inputs=allow_cpu_inputs):
        _summary.fallback(node, "torch._inductor.lowering.fallback_node_due_to_unsupported_type")
        return True

    if not isinstance(node.target, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        return False

    if not _LoweringGuard.has(node.target):
        _summary.fallback(node, "not in lowering whitelist")
        return True

    in_nodes = [n for n in pytree.arg_tree_leaves(*node.args, **node.kwargs) if isinstance(n, torch.fx.Node)]
    in_metas = get_node_meta(in_nodes)
    out_metas = get_node_meta([node])

    for meta in itertools.chain(in_metas, out_metas):
        if not isinstance(meta, torch._subclasses.FakeTensor):
            continue

        for expr in itertools.chain(meta.shape, meta.stride()):
            if any([v in str(expr) for v in ('ModularIndex', 'Max', 'Min')]):
                _summary.fallback(node, f"unsupported shape/stride expr {expr} with ['ModularIndex', 'Max', 'Min']")
                return True

        if meta.stride() and all([str(v) != "1" for v in meta.stride()]):
            _summary.fallback(node, f"performance of non-contiguous stride {meta.stride()} maybe worse than eager")
            return True

    support_input_dtypes, support_output_dtypes = _LoweringGuard.dtypes_support(node.target)
    for meta in in_metas:
        if isinstance(meta, torch._subclasses.FakeTensor) and meta.dtype not in support_input_dtypes:
            _summary.fallback(node, f"input dtype {meta.dtype} not in {support_input_dtypes}")
            return True

    for meta in out_metas:
        if isinstance(meta, torch._subclasses.FakeTensor) and meta.dtype not in support_output_dtypes:
            _summary.fallback(node, f"output dtype {meta.dtype} not in {support_output_dtypes}")
            return True

    _summary.lowered(node)
    return False


def _stub_debugging_host_only():
    import logging
    logger.setLevel(logging.DEBUG)
    from inductor_npu_ext import config
    config._debugging_host_only = True
    from torch._inductor import config as inductor_config
    inductor_config.compile_threads = 1
    inductor_config.force_disable_caches = True
    register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)
