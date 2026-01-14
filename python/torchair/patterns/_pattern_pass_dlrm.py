__all__ = []

import logging
import operator

import torch
import torchair

from torch.fx import Node
from torchair.core.utils import logger


def _optimize_dlrm_pattern(gm: torch.fx.GraphModule, example_inputs, 
    config: torchair.CompilerConfig):
    fx_graph = gm.graph
    # Part 1: Optimize embedding layer pattern
    _optimize_embedding_layer_pattern(fx_graph)
    # Part 2: Optimize triangular matrix operations (simplify ones_like+triu+where to tril)
    _optimize_triangular_matrix_pattern(fx_graph)
    # Part 3: Optimize predict_layers pattern
    _optimize_predict_layers_pattern(fx_graph)
    # Recompile the graph module
    gm.recompile()
    logger.debug("All optimizations completed")
    return gm


def _replace_arg(args, node, replace_node):
    if isinstance(args, (list, tuple)):
        new_args = []
        for arg in args:
            if arg == node:
                new_args.append(replace_node)
            elif isinstance(arg, (list, tuple)):
                new_args.append(_replace_arg(arg, node, replace_node))
            else:
                new_args.append(arg)
        return type(args)(new_args)
    elif args == node:
        return replace_node
    else:
        return args


def _optimize_predict_layers_pattern(fx_graph):
    """
    Optimize predict_layers pattern: Split a single large matrix multiplication into two 
    independent matrix multiplications then add them
    Handle both addmm and linear cases
    """
    logger.debug("Starting to optimize predict_layers operations...")                      
    # Find linear layer nodes (could be addmm or linear)
    for node in fx_graph.nodes:
        is_addmm = (node.op == "call_function" and 
                   hasattr(node.target, '_name') and 
                   node.target._name == "aten::addmm")
        is_linear = (node.op == "call_function" and
                    hasattr(node.target, '_name') and 
                    node.target._name == "aten::linear")
        if not (is_addmm or is_linear):
            continue
        logger.debug("Found linear layer node: %s", node)
        # By checking if the input comes from concat([dense_embedding, interaction_output])
        if len(node.args) < 3:
            continue
        if is_addmm:
            # addmm format: addmm(bias, input, weight.t())
            bias_node = node.args[0]
            input_node = node.args[1]
            weight_t_node = node.args[2] 
        else:  
            # linear format: linear(input, weight, bias)
            input_node = node.args[0]
            weight_node = node.args[1]
            bias_node = node.args[2] if len(node.args) > 2 else None
        # Check if input is a concat operation
        if (isinstance(input_node, Node) and
            input_node.op == "call_function" and
            hasattr(input_node.target, '_name') and
            input_node.target._name == "aten::cat"):
            logger.debug("Found concat input node")
            # Check concat parameters
            if (len(input_node.args) >= 2 and
                isinstance(input_node.args[0], (list, tuple)) and
                len(input_node.args[0]) == 2):
                concat_inputs = input_node.args[0]
                dense_embedding_node = concat_inputs[0]
                interaction_output_node = concat_inputs[1]
                logger.debug("dense_embedding_node: %s", dense_embedding_node)
                logger.debug("interaction_output_node: %s", interaction_output_node)
                if is_addmm:
                    # Find the transpose node of the weight
                    if (isinstance(weight_t_node, Node) and
                        weight_t_node.op == "call_function" and
                        hasattr(weight_t_node.target, '_name') and
                        weight_t_node.target._name == "aten::t"):
                        weight_node = weight_t_node.args[0]
                    else:
                        logger.debug("Cannot find original weight node, skipping optimization")
                        continue
                if not isinstance(weight_node, Node):
                    logger.debug("Weight is not a node, skipping")
                    continue
                # Get embedding_size (inferred from dense_embedding's shape)
                embedding_size = None
                if (hasattr(dense_embedding_node, 'meta') and 
                    'val' in dense_embedding_node.meta and
                    len(dense_embedding_node.meta['val'].shape) >= 2):
                    embedding_size = dense_embedding_node.meta['val'].shape[1]
                    logger.debug("Inferred embedding_size %s", embedding_size)
                if embedding_size is None:
                    logger.debug("Cannot infer embedding_size, trying to infer from concat input dimensions")
                    # If cannot get from meta, try to infer from concat input
                    if (hasattr(input_node, 'meta') and 
                        'val' in input_node.meta and
                        len(input_node.meta['val'].shape) >= 2 and
                        hasattr(dense_embedding_node, 'meta') and 
                        'val' in dense_embedding_node.meta and
                        len(dense_embedding_node.meta['val'].shape) >= 2):
                        total_size = input_node.meta['val'].shape[1]
                        dense_size = dense_embedding_node.meta['val'].shape[1]
                        interaction_size = total_size - dense_size
                        embedding_size = dense_size
                        logger.debug("Inferred from concat: total_size=%s, " + 
                        "dense_size=%s, interaction_size=%s", total_size, dense_size, interaction_size)
                if embedding_size is None:
                    logger.debug("Cannot infer embedding_size, skipping optimization")
                    continue  
                # Perform optimization
                with fx_graph.inserting_before(node):
                    weight_emb_node = fx_graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        args=(weight_node, 1, 0, embedding_size)
                    )
                    weight_act_node = fx_graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        args=(weight_node, 1, embedding_size, 9223372036854775807)  # 9223372036854775807 is the end
                    )
                    if is_addmm:
                        weight_emb_t_node = fx_graph.call_function(
                            torch.ops.aten.t.default,
                            args=(weight_emb_node,)
                        )
                        weight_act_t_node = fx_graph.call_function(
                            torch.ops.aten.t.default,
                            args=(weight_act_node,)
                        )
                        output1_node = fx_graph.call_function(
                            torch.ops.aten.mm.default,
                            args=(dense_embedding_node, weight_emb_t_node)
                        )
                        output2_node = fx_graph.call_function(
                            torch.ops.aten.mm.default,
                            args=(interaction_output_node, weight_act_t_node)
                        )
                    else:
                        # For linear, use matmul directly
                        output1_node = fx_graph.call_function(
                            torch.ops.aten.matmul.default,
                            args=(dense_embedding_node, weight_emb_node)
                        )
                        output2_node = fx_graph.call_function(
                            torch.ops.aten.matmul.default,
                            args=(interaction_output_node, weight_act_node)
                        )
                    output_sum_node = fx_graph.call_function(
                        torch.ops.aten.add.Tensor,
                        args=(output1_node, output2_node)
                    )
                    final_output_node = output_sum_node
                    if bias_node is not None:
                        final_output_node = fx_graph.call_function(
                            torch.ops.aten.add.Tensor,
                            args=(output_sum_node, bias_node)
                        )
                # Redirect all users of the original linear layer node to the new output node
                node_users = list(node.users.keys())
                for user_node in node_users:
                    if hasattr(user_node, 'args') and user_node.args:
                        user_node.args = _replace_arg(user_node.args, node, final_output_node)
                    if hasattr(user_node, 'kwargs') and user_node.kwargs:
                        user_node.kwargs = _replace_arg(user_node.kwargs, node, final_output_node)
                # Collect nodes to remove
                nodes_to_remove = {node, input_node}
                if is_addmm and hasattr(weight_t_node, 'users') and len(weight_t_node.users) == 0:
                    nodes_to_remove.add(weight_t_node)
                _safe_remove_nodes(fx_graph, nodes_to_remove)
                logger.debug("predict_layers optimization completed")
                return  # Only optimize one pattern at a time


# Keep the original helper functions unchanged
def _optimize_triangular_matrix_pattern(fx_graph):
    """
    Optimize triangular matrix operations: 
    Simplify ones_like+triu+where pattern to tril operation
    """
    logger.debug("Starting to optimize triangular matrix operations...")
    # Find matching pattern: ones_like -> triu -> to(bool) -> where(zeros_like, xactions)
    for node in fx_graph.nodes:
        if (node.op == "call_function" and
            hasattr(node.target, '_name') and 
            node.target._name == "aten::where.self"):
            # Check where node parameters
            if len(node.args) >= 3:
                condition_node = node.args[0]
                true_value_node = node.args[1] 
                false_value_node = node.args[2]
                # Check if condition is triu(ones_like(...)).to(bool)
                if (isinstance(condition_node, Node) and
                    condition_node.op == "call_function" and
                    hasattr(condition_node.target, '_name') and
                    condition_node.target._name == 'npu::_npu_dtype_cast' and
                    len(condition_node.args) >= 2 and
                    condition_node.args[1] == torch.bool):
                    triu_node = condition_node.args[0]
                    if (isinstance(triu_node, Node) and
                        triu_node.op == "call_function" and
                        triu_node.target._name == "aten::triu"):
                        ones_like_node = triu_node.args[0]
                        if (isinstance(ones_like_node, Node) and
                            ones_like_node.op == "call_function" and
                            ones_like_node.target._name == "aten::ones_like"):
                            # Check if true_value is zeros_like
                            if (isinstance(true_value_node, Node) and
                                true_value_node.op == "call_function" and
                                true_value_node.target._name == "aten::zeros_like"):
                                # Check if false_value is xactions (should be same as ones_like input)
                                xactions_node = false_value_node
                                ones_like_input = ones_like_node.args[0]
                                zeros_like_input = true_value_node.args[0]
                                # Ensure ones_like and zeros_like inputs are the same, and same as xactions
                                if (ones_like_input == zeros_like_input and 
                                    ones_like_input == xactions_node):
                                    logger.debug("Found triangular matrix optimization pattern")
                                    # Replace entire pattern with tril
                                    with fx_graph.inserting_before(node):
                                        # Create tril node, diagonal=-1
                                        tril_node = fx_graph.call_function(
                                            torch.ops.aten.tril.default,
                                            args=(xactions_node, -1)
                                        )
                                    # Redirect all users of where node to tril node
                                    node_users = list(node.users.keys())
                                    for user_node in node_users:
                                        if hasattr(user_node, 'args') and user_node.args:
                                            user_node.args = _replace_arg(user_node.args, node, tril_node)
                                    # Collect nodes to remove
                                    nodes_to_remove = {node, condition_node, triu_node, ones_like_node, true_value_node}
                                    # Safely remove nodes
                                    _safe_remove_nodes(fx_graph, nodes_to_remove)
                                    logger.debug("Triangular matrix optimization completed")
                                    return  # Only optimize one pattern at a time


def _optimize_embedding_layer_pattern(fx_graph):
    """
    Optimize embedding layer pattern: 
    Convert loop-unrolled slice/embedding/sum operations to split/cat/sum pattern
    """
    # Identify pattern - find all slice, embedding and sum nodes
    slice_nodes = []
    embedding_nodes = []
    sum_nodes = []
    cat_node = None
    for node in fx_graph.nodes:
        if node.op == "call_function":
            # Identify slice nodes (used to split feat_ids)
            if (hasattr(node.target, '_name') and 
                node.target._name == 'aten::slice.Tensor' and
                len(node.args) >= 4 and node.args[1] == 1):  # slice on dim=1
                slice_nodes.append(node)
            # Identify embedding nodes
            elif (hasattr(node.target, '_name') and 
                  node.target._name == 'aten::embedding'):
                embedding_nodes.append(node)
            # Identify sum nodes (sum on dim=1)
            elif (hasattr(node.target, '_name') and 
                  node.target._name == 'aten::sum.dim_IntList' and
                  len(node.args) >= 2 and node.args[1] == [1]):
                sum_nodes.append(node)
            # Identify cat node (concat on dim=1)
            elif (hasattr(node.target, '_name') and 
                  node.target._name == 'aten::cat' and
                  len(node.args) >= 2 and node.args[1] == 1):
                # Check if cat inputs are all sum nodes
                if (len(node.args) > 0 and isinstance(node.args[0], (list, tuple)) and
                    all(isinstance(arg, Node) and arg in sum_nodes for arg in node.args[0])):
                    cat_node = node
    # If pattern is not complete, return directly
    if len(slice_nodes) < 2 or len(embedding_nodes) < 2 or len(sum_nodes) < 2 or not cat_node:
        logger.debug("Pattern incomplete: slice_nodes=%s, " + 
         "embedding_nodes=%s, sum_nodes=%s, " +
         "cat_node=%s", len(slice_nodes), len(embedding_nodes), len(sum_nodes), cat_node)
        return 
    logger.debug("Found optimization pattern: %s slices, " + 
    "%s embeddings, %s sums, 1 cat", len(slice_nodes), len(embedding_nodes), len(sum_nodes))
    # Analyze slice node patterns, extract group_size information
    slice_patterns = []
    for slice_node in slice_nodes:
        if len(slice_node.args) >= 4:
            start = slice_node.args[2]
            end = slice_node.args[3]
            if slice_node.args[1] == 1:
                slice_patterns.append((slice_node, start, end))
    slice_patterns.sort(key=lambda x: x[1])
    if len(slice_patterns) < 2:
        logger.debug("Insufficient slice patterns")
        return 
    # Calculate group_size (difference between adjacent slice starts)
    group_size = slice_patterns[1][1] - slice_patterns[0][1]
    logger.debug("Detected group_size: %s", group_size)
    # Reconstruct computation graph
    feat_ids_node = None
    embedding_weight_node = None
    # Find feat_ids input from first slice node
    first_slice = slice_patterns[0][0]
    if hasattr(first_slice, 'args') and len(first_slice.args) > 0:
        # Trace back to find original feat_ids input
        input_node = first_slice.args[0]
        while (isinstance(input_node, Node) and input_node.op == "call_function" and 
               hasattr(input_node.target, '_name') and 
               input_node.target._name == 'aten::slice.Tensor'):
            if len(input_node.args) > 0:
                input_node = input_node.args[0]
            else:
                break
        feat_ids_node = input_node
    # Find weight from first embedding node
    if embedding_nodes and len(embedding_nodes[0].args) > 0:
        embedding_weight_node = embedding_nodes[0].args[0]
    if not feat_ids_node or not embedding_weight_node:
        logger.debug("Cannot find feat_ids or embedding_weight node")
        return 
    logger.debug("feat_ids_node: %s, embedding_weight_node: %s", feat_ids_node, embedding_weight_node)
    # Insert new optimized operations
    with fx_graph.inserting_before(slice_patterns[0][0]):
        num_slices = len(slice_patterns)
        split_sizes = [group_size] * num_slices
        if slice_patterns:
            last_start = slice_patterns[-1][1]
            last_end = slice_patterns[-1][2]
            if last_end == 9223372036854775807:  # Maximum value, meaning to the end
                # Need to calculate actual size
                if hasattr(feat_ids_node, 'meta') and 'val' in feat_ids_node.meta:
                    total_size = feat_ids_node.meta['val'].shape[1]
                    last_size = total_size - last_start
                    split_sizes[-1] = last_size
                else:
                    # Default to group_size
                    pass
        logger.debug("Using split_sizes: %s", split_sizes)
        split_node = fx_graph.call_function(
            torch.ops.aten.split_with_sizes.default, 
            args=(feat_ids_node, split_sizes, 1)
        ) 
        # Apply embedding and sum to each split result
        embedding_results = []
        for i in range(num_slices):
            getitem_node = fx_graph.call_function(
                operator.getitem,
                args=(split_node, i)
            )
            # Embedding operation
            embedding_node = fx_graph.call_function(
                torch.ops.aten.embedding.default,
                args=(embedding_weight_node, getitem_node)
            )
            embedding_results.append(embedding_node)
        # Concat all embedding results on dim=2
        cat_embedding_node = fx_graph.call_function(
            torch.ops.aten.cat.default,
            args=(embedding_results, 2)  # Concat on dim=2
        )
        # Perform one sum operation (on dim=1)
        new_sum_node = fx_graph.call_function(
            torch.ops.aten.sum.dim_IntList,
            args=(cat_embedding_node, [1])
        )
    cat_users = list(cat_node.users.keys())
    for user_node in cat_users:
        if hasattr(user_node, 'args'):
            user_node.args = _replace_arg(user_node.args, cat_node, new_sum_node)
    nodes_to_remove = set()
    nodes_to_remove.update(slice_nodes)
    nodes_to_remove.update(embedding_nodes)
    nodes_to_remove.update(sum_nodes)
    nodes_to_remove.add(cat_node)
    _safe_remove_nodes(fx_graph, nodes_to_remove)
    # Recompile graph module  
    logger.debug("Embedding layer optimization completed")


def _safe_remove_nodes(fx_graph, nodes_to_remove):
    # First create a list of nodes to delete, sorted in reverse topological order
    all_nodes = list(fx_graph.nodes)
    sorted_nodes_to_remove = []
    for node in reversed(all_nodes):
        if node in nodes_to_remove:
            sorted_nodes_to_remove.append(node)
    for node in sorted_nodes_to_remove:
        # Only delete if node has no users
        if len(node.users) == 0:
            try:
                fx_graph.erase_node(node)  
                logger.debug("Successfully deleted node: %s", node)
            except Exception as e:
                logger.debug("Error deleting node %s: %s", node, e)
        else:
            logger.debug("Node %s still has %s users, skipping deletion", node, len(node.users))