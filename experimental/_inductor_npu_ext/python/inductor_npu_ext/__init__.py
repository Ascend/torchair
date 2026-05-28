# pylint: disable=W0143,W1203,R1729
import sys
import itertools

import torch
import torch.utils._pytree as pytree
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor import graph as torch__inductor_graph

from inductor_npu_ext.npu import NPUScheduling, NpuWrapperCodeGen
from inductor_npu_ext.common import logger, current_soc, Soc
from inductor_npu_ext.common.utils import patch_fn
from inductor_npu_ext.config import _debug_options

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
    from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401
    from torch._dynamo import config as dynamo_config

    dynamo_config.recompile_limit = 16  # default 8 is always not enough for per-layer compile
    dynamo_config.specialize_float = True  # enable float specialization until launch with scalar supported
    from torch._inductor import config as inductor_config

    inductor_config.unroll_reductions_threshold = 1  # disable unroll reductions
    inductor_config.size_asserts = (
        False  # npu ops always return contiguous tensors which maybe different from meta outputs
    )
    from torch._inductor.runtime.hints import DeviceProperties

    DeviceProperties.multi_processor_count = 0  # disable multi processor count


def _load_npu_passes():
    from torch._inductor import config as inductor_config
    from inductor_npu_ext.passes.auto_functionalize_legacy_ops import auto_functionalize_legacy_ops

    if inductor_config.pre_grad_custom_pass is None:
        inductor_config.pre_grad_custom_pass = auto_functionalize_legacy_ops
    else:
        if inductor_config.pre_grad_custom_pass != auto_functionalize_legacy_ops:
            logger.warning(
                f"`pre_grad_custom_pass` has already been registered as {inductor_config.pre_grad_custom_pass}, "
                f"registration of `auto_functionalize_legacy_ops` will be skipped. "
                f"Note that this may introduce extra tensormove operations. "
                f"You can import `auto_functionalize_legacy_ops` and call it within your custom pass: "
                f"`from inductor_npu_ext.passes.auto_functionalize_legacy_ops import auto_functionalize_legacy_ops`"
            )


def _finetune_decompose():
    """清掉 inductor 的 decomposition 表，只留少量必需项。

    保留的 op 都是 inductor 没注册直接 lowering、必须靠 decomposition 才能下沉
    到 NPUOverrides 支持的原子 op 的"复合 op"：
      - _to_copy：dtype cast 路径
      - silu(x) = x / (1 + (-x).exp())：让 silu 拆成 neg+exp+add+div
      - sgn(x) → sign(x)（实数路径），让 sgn 走 NPUOverrides.sign → ir.sign
      - floor_divide：inductor 没注册直接 lowering，靠 decomposition 拆成
        div + floor，两者都已支持
    """
    from torch._inductor.decomposition import decompositions

    def _packet_overloads(packet):
        return {getattr(packet, ov) for ov in packet.overloads()}

    keep = _packet_overloads(aten._to_copy)
    keep |= _packet_overloads(aten.silu)
    keep |= _packet_overloads(aten.sgn)
    keep |= _packet_overloads(aten.detach)
    keep |= _packet_overloads(aten.floor_divide)
    decompositions_whitelist = {k: v for k, v in decompositions.items() if k in keep}
    decompositions.clear()
    decompositions.update(decompositions_whitelist)


def _finetune_lowering():
    import inductor_npu_ext.lowering.aten_lowering  # noqa: F401, make sure lowering rules for npu ops are registered  # typos:ignore


def _load_npu_lowering():
    import inductor_npu_ext.lowering.npu_lowering  # noqa: F401, make sure lowering rules for npu ops are registered  # typos:ignore


_finetune_inductor_config()
_finetune_lowering()


if "decompose" not in _debug_options:
    _finetune_decompose()

if "cpu" not in _debug_options:
    _load_npu_lowering()
    _load_npu_passes()
else:
    register_backend_for_device("cpu", NPUScheduling, NpuWrapperCodeGen)


@patch_fn(torch__inductor_graph, "fallback_node_due_to_unsupported_type")
def _fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True, *, orig_fn=None):
    from inductor_npu_ext.lowering.common import _LoweringGuard, _summary
    from inductor_npu_ext.common.utils import get_node_meta

    if orig_fn(node, allow_cpu_inputs=allow_cpu_inputs):
        _summary.fallback(node, "torch._inductor.lowering.fallback_node_due_to_unsupported_type")
        return True

    if "lowering" in _debug_options:
        _summary.lowered(node)
        return False

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
            if current_soc and current_soc < Soc.A5:
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
