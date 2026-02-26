import torch
import torch.library
import torch_npu

from inductor_npu_ext.common import logger

modern = torch.library.Library("modern", "FRAGMENT")
modern.define("npu_scatter_nd_update_(Tensor(a!) x, Tensor indices, Tensor updates) -> None")


@torch.library.impl("modern::npu_scatter_nd_update_", "NPU")
def npu_scatter_nd_update_impl(x, indices, updates):
    torch.ops.npu.npu_scatter_nd_update_(x, indices, updates)


@torch.library.impl("modern::npu_scatter_nd_update_", "Meta")
def npu_scatter_nd_update_meta(x, indices, updates):
    pass


def _replace_legacy_npu_scatter_nd_update_(graph, node):
    logger.debug(f"Replacing legacy npu_scatter_nd_update_ at node {node.name}")
    modern_op = torch.ops.modern.npu_scatter_nd_update_
    with graph.inserting_after(node):
        graph.call_function(modern_op, args=node.args, kwargs=node.kwargs)
        node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)


def auto_functionalize_legacy_ops(graph: torch.fx.graph.Graph):
    """
    Registered @pre_grad_custom_pass

    Replaces legacy in-place NPU ops such as torch.ops.npu.npu_scatter_nd_update_ in FX graph
    with a modern, custom operator. The modern operator can be auto-functionalized by AOT,
    and can be re-inplace after inductor fuse.
    """

    for node in list(graph.nodes):
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.npu.npu_scatter_nd_update_:
            _replace_legacy_npu_scatter_nd_update_(graph, node)

    graph.lint()
    return graph
