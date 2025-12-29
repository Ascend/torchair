__all__ = []

import functools
import logging
import os
import warnings
from typing import List, Optional, Callable, Any, Dict, Tuple, Union

import torch
from torch import fx
from torch.fx import Graph, GraphModule, Node

from torchair.core.utils import logger


def add_stream_label_to_node_meta(graph_module: torch.fx.GraphModule) -> None:
    """
    Add stream labels to nodes in the FX graph based on their scope.
    - Nodes within a stream scope: meta["stream_label"] = corresponding stream label
    - Nodes outside a stream scope: meta["stream_label"] = None (default stream)

    Args:
        graph_module (torch.fx.GraphModule): The FX graph module to set stream labels

    Note: Those stream labels set by this interface are temporary tag.
          Please proactively call this interface to set for all nodes before using stream_label attr in node meta.
    """

    scope_enter_nodes_stack = []
    current_stream = None

    for node in graph_module.graph.nodes:
        if not hasattr(node, 'meta'):
            # Some node may not have 'meta' attr, such as int placeholder
            node.meta = {}

        if str(node.target) == "air.scope_enter.default":
            is_user_stream = len(node.args) > 0 and '_user_stream_label' in node.args[0]
            current_stream = node.args[1][0] if is_user_stream and len(node.args) > 1 else None
            node.meta["stream_label"] = current_stream
            scope_enter_nodes_stack.append(node)

        elif str(node.target) == "air.scope_exit.default":
            node.meta["stream_label"] = current_stream
            if scope_enter_nodes_stack:
                scope_enter_nodes_stack.pop()
                current_stream = scope_enter_nodes_stack[-1].args[1][0] if scope_enter_nodes_stack and len(
                    scope_enter_nodes_stack[-1].args) > 1 else None

        else:
            node.meta["stream_label"] = current_stream

    logger.debug('End to add stream labels to all FX nodes in graph %s', id(graph_module))
    graph_module.graph.lint()


def verify_nodes_on_same_stream(nodes: List[torch.fx.Node]) -> bool:
    """
    Verify that all input nodes are on the same stream

    Args:
        nodes (List[torch.fx.Node]): The nodes to verify stream labels
    """

    all_streams = []

    for node in nodes:
        if not hasattr(node, 'meta') or 'stream_label' not in node.meta:
            logger.debug("There is some node without steam label, the verification interface return False. "
                         "Current node is %s.", node.target)
            return False

        current_stream = node.meta.get("stream_label")
        all_streams.append(current_stream)

    if len(set(all_streams)) > 1:
        logger.debug("There is more than one stream in the input nodes, the verification interface return False."
                     "All nodes and stream labels are %s.", dict(zip(nodes, all_streams)))
        return False

    return True


def _is_inplace_node(node: Node):
    # simple analysis of function schema to determine
    # if this is an inplace variant, it might not
    # be entirely correct, but it's good enough for now.
    if isinstance(node.target, str):
        return node.target.endswith("_")
    elif hasattr(node.target, 'overloadpacket') and hasattr(node.target.overloadpacket, '__name__'):
        return node.target.overloadpacket.__name__.endswith("_")
    else:
        return False


def _find_mutated_user_inputs(gm: fx.GraphModule):
    from torchair._acl_concrete_graph.graph_pass import inplaceable_npu_ops
    inplace_ops_mutated_indices = {inplaceable_op.inplace_op: inplaceable_op.mutated_arg
                                   for _, inplaceable_op in inplaceable_npu_ops.items()}

    inplace_node_args_list = []
    placeholder_args = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_args.add(node.target)
        elif _is_inplace_node(node):
            inplace_node_args_list.append(node.args[0].name)
        elif node.target in inplace_ops_mutated_indices:
            for idx in inplace_ops_mutated_indices[node.target]:
                inplace_node_args_list.append(node.args[idx].name)

    return [arg for arg in inplace_node_args_list if arg in placeholder_args]