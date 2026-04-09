import sys
import itertools

import torch
import torch.utils._pytree as pytree
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor import graph as torch__inductor_graph

from inductor_npu_ext.npu import NPUScheduling, NpuWrapperCodeGen
from inductor_npu_ext.common import logger
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
    from torch._inductor.codegen import cpu_device_op_overrides
    from torch._dynamo import config as dynamo_config
    dynamo_config.recompile_limit = 16  # default 8 is always not enough for per-layer compile
    dynamo_config.specialize_float = True  # enable float specialization until launch with scalar supported
    from torch._inductor import config as inductor_config
    inductor_config.unroll_reductions_threshold = 1  # disable unroll reductions
    inductor_config.size_asserts = False  # npu ops always return contiguous tensors which maybe different from meta outputs
    from torch._inductor.runtime.hints import DeviceProperties
    DeviceProperties.multi_processor_count = 0  # disable multi processor count


def _load_npu_passes():
    from torch._inductor import config as inductor_config
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


def _finetune_decompose():
    from torch._inductor.decomposition import decompositions
    decompositions_whitelist = {
        k: v for k, v in decompositions.items() if k in {aten._to_copy.default}
    }
    decompositions.clear()
    decompositions.update(decompositions_whitelist)


def _finetune_lowering():
    import inductor_npu_ext.lowering.aten_lowering  # noqa: F401, make sure lowering rules for npu opps are registered


def _load_npu_lowering():
    import inductor_npu_ext.lowering.npu_lowering  # noqa: F401, make sure lowering rules for npu opps are registered


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
