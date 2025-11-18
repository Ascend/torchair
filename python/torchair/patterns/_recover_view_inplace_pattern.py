__all__ = []

import operator
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Any, Dict, Tuple, Union, Set
import logging
import os
import sys

import torch
from torch.fx import GraphModule, Node, Proxy
from torch.fx.node import Argument, Target
from torch.utils._mode_utils import no_dispatch
from torch._dynamo.utils import detect_fake_mode
from torch._subclasses.fake_tensor import is_fake

from torchair.core.utils import logger
from torchair.configs.compiler_config import CompilerConfig


@dataclass
class InplaceOpInfo:
    inplace_op: Callable
    output_to_ref_input_idx_mapping: Dict[int, int]


_INPLACE_OPS_MAP = {}

if hasattr(torch.ops.npu, "scatter_update"):
    _INPLACE_OPS_MAP.update({
        torch.ops.npu.scatter_update.default: InplaceOpInfo(
            inplace_op=torch.ops.npu.scatter_update_.default,
            output_to_ref_input_idx_mapping={0: 0})
    })

if hasattr(torch.ops.npu, "npu_scatter_nd_update"):
    _INPLACE_OPS_MAP.update({
        torch.ops.npu.npu_scatter_nd_update.default: InplaceOpInfo(
            inplace_op=torch.ops.npu.npu_scatter_nd_update_.default,
            output_to_ref_input_idx_mapping={0: 0})
    })

if hasattr(torch.ops.npu, "npu_quant_scatter"):
    _INPLACE_OPS_MAP.update({
        torch.ops.npu.npu_quant_scatter.default: InplaceOpInfo(
            inplace_op=torch.ops.npu.npu_quant_scatter_.default,
            output_to_ref_input_idx_mapping={0: 0})
    })


def _get_other_to_args_copy_nodes(graph_module: GraphModule) -> List[Node]:
    graph = graph_module.graph

    all_other_to_args_copy_nodes = []
    for node in graph.nodes:
        if not (node.op == "call_function" and node.target == torch.ops.aten.copy_.default):
            continue
        if len(node.args) != 2:
            # undefined copy_ ops which inputs num is not 2, skip it.
            logger.debug("Skip copy_ node: %s, due to unexpected input num %s, expected 2 inputs.",
                         node, len(node.args))
            continue
        if not (node.args[0].op == "placeholder" and node.args[1].op != "placeholder"):
            # copy for inplace op must be from other op to placeholder
            logger.debug("Skip copy_ node: %s, expect copy from other_node to placeholder_node, but get an "
                         "copy_ node from %s to %s.", node, node.args[1].op, node.args[0].op)
            continue
        if hasattr(node.args[0], "meta") and hasattr(node.args[1], "meta"):
            dst_shape = node.meta['val'].size()
            src_shape = node.meta['val'].size()
            if dst_shape != src_shape:
                # copy for inplace op must be of the same shape，broadcasting is unexpected.
                logger.debug("Skip copy_ node: %s, expect the shape of src and dst to be the same, but src shape "
                             "is %s, and dst shape is %s.", node, src_shape, dst_shape)
                continue
            all_other_to_args_copy_nodes.append(node)

    logger.debug('In fx_graph, find all copy_ nodes to optimize, those are %s', all_other_to_args_copy_nodes)
    return all_other_to_args_copy_nodes


def find_optimization_pattern(graph_module: GraphModule) -> List:
    """
    Find those specific computational patterns in the graph module that can be converted to inplace operations.

    Args:
        graph_module (torch.fx.GraphModule): The FX graph module to find patterns

    Returns:
        List of matched patterns where each pattern contains non_inplace nodes that can be optimized.


    Target pattern:  input -> view -> non_inplace_op -> view -> copy_
    """

    all_copy_nodes = _get_other_to_args_copy_nodes(graph_module)

    matched_pattern = []
    for copy_node in all_copy_nodes:
        post_view = copy_node.args[1]
        if not (post_view.op == "call_function" and post_view.target == torch.ops.aten.view.default):
            # just find case: data->view->non_inplace->view->copy_
            logger.debug("Skip copy_ node: %s, because src input of copy is not expected view node, but is %s.",
                         copy_node, post_view)
            continue
        non_inplace_node = post_view.args[0]
        if not (non_inplace_node.op == "call_function" and non_inplace_node.target in _INPLACE_OPS_MAP.keys()):
            # just find non_inplace op that can be transfer to inplace op
            logger.debug("Skip copy_ node: %s, because input of post_view is not in expected non_inplace "
                         "nodes set %s, but is %s.", copy_node, _INPLACE_OPS_MAP.keys(), non_inplace_node)
            continue
        pre_view_index = _INPLACE_OPS_MAP[non_inplace_node.target].output_to_ref_input_idx_mapping[0]
        pre_view = non_inplace_node.args[pre_view_index]
        if not (pre_view.op == "call_function" and pre_view.target == torch.ops.aten.view.default):
            # pre op before non_inplace must be view
            logger.debug("Skip copy_ node: %s, because ref input of non_inplace node is not expected view node, "
                         "but is %s.", copy_node, pre_view)
            continue
        if copy_node.args[0] != pre_view.args[0]:
            # pre_view input_0 must be copy input_0
            logger.debug("Skip copy_ node: %s, because dst input of copy and input of pre_view are not the same node, "
                         "those are %s and %s.", copy_node, copy_node.args[0], pre_view.args[0])
            continue
        if not (len(pre_view.users) == 1 and len(pre_view.args[0].users) == 2):
            # pre_view between data and non_inplace, must be used by only one op.
            # data op must be used by pew_view and copy
            logger.debug("Skip copy_ node: %s, pre_view node must have only one user， but get %s, "
                         " and placeholder node must have only two user, but get %s.",
                         copy_node, len(pre_view.users), len(pre_view.args[0].users))
            continue

        matched_pattern.append({"placeholder": pre_view.args[0],
                                "pre_view": pre_view,
                                "non_inplace": non_inplace_node,
                                "post_view": post_view,
                                "copy_": copy_node,
                                })

    logger.info('In fx_graph, find %s matched patterns for view inplace optimization, '
                'those are  %s.', len(matched_pattern), matched_pattern)
    return matched_pattern


def _is_supported_cann_version(target_version):
    is_supported_cann_version = False
    try:
        from torch_npu.npu.utils import _is_gte_cann_version
        is_supported_cann_version = _is_gte_cann_version(target_version, module="CANN")
    except Exception as err:
        logger.warning_once(f"Failed to import _is_gte_cann_version from torch_npu, skip cann version check.")

    return is_supported_cann_version


def recover_view_inplace_pattern(graph_module: torch.fx.GraphModule, example_inputs=None, config: CompilerConfig = None) -> GraphModule:
    """
    This function identifies specific computational patterns
    and replaces non-inplace operations enerated during the functionalization with
    their corresponding inplace versions to reduce memory allocation and improve execution efficiency.

    Args:
        graph_module (torch.fx.GraphModule): The FX graph module to be optimized

    Returns:
        torch.fx.GraphModule: The optimized graph module,
                              in which the matched patterns have been replaced by inplace operations

    Target pattern and replacement examples:
    * ***********************************************************************************************
    *            placeholder                        placeholder
    *              |       \                            |
    *           pre_view    \                        pre_view
    *              |         \                          |
    *       scatter_update   |        --->        scatter_update_
    *              |        /                           |
    *         post_view    /                        post_view
    *            /  \     /                             |
    *           /    \   /                              |
    *      other_op  copy_                          other_op
    * ***********************************************************************************************
    """

    # Note: This pass may cause format propagation errors in GE graph, resulting in unexpected NCHW.
    # Therefore, it was only enabled in versions after 8.3.RC1, and the related errors have been fixed since then.
    if not _is_supported_cann_version("8.3.RC1"):
        logger.info(f"Skip running the current pattern of recovering to view inplace operators, "
                    f"due to unsupported cann version, please update to at least version 8.3.RC1.")
        return graph_module

    matched_patterns = find_optimization_pattern(graph_module)

    graph = graph_module.graph
    for pattern in matched_patterns:
        placeholder = pattern["placeholder"],
        pre_view = pattern["pre_view"]
        non_inplace_op = pattern["non_inplace"]
        post_view = pattern["post_view"]
        copy_ = pattern["copy_"]

        # insert inplace op to replace original non_inplace op
        with graph.inserting_after(non_inplace_op):
            inplace_op = graph.call_function(_INPLACE_OPS_MAP[non_inplace_op.target].inplace_op,
                                             non_inplace_op.args, non_inplace_op.kwargs)
            inplace_op.name = f"{non_inplace_op.name}_inplace"

        # update non_inplace op users
        non_inplace_op.replace_all_uses_with(inplace_op, propagate_meta=True)
        logger.debug('In %s, success to replace non_inplace node %s by inserting new inplace node %s[target: %s].',
                     "ACL graph" if config.mode.value == "reduce-overhead" else "GE graph",
                     non_inplace_op, inplace_op, inplace_op.target)

        # delete old copy node and non_inplace node
        graph.erase_node(copy_)
        graph.erase_node(non_inplace_op)

    # lint and recompile for graph
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
