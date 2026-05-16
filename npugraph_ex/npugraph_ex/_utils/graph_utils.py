__all__ = []

import functools
import logging
import time
from collections import Counter
from typing import List, Optional, Callable, Dict

import torch
from torch import fx
from torch.fx import GraphModule, Node

from npugraph_ex.core.utils import logger


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
            scope_enter_nodes_stack.append(current_stream)
            is_user_stream = len(node.args) > 0 and '_user_stream_label' in node.args[0]
            if is_user_stream:
                current_stream = node.args[1][0] if len(node.args) > 1 else None
            node.meta["stream_label"] = current_stream

        elif str(node.target) == "air.scope_exit.default":
            node.meta["stream_label"] = current_stream
            if scope_enter_nodes_stack:
                current_stream = scope_enter_nodes_stack.pop()

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
    # No need to check nodes_on_same_stream if there is only one node.
    if len(nodes) < 2:
        return True

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
    from npugraph_ex._acl_concrete_graph.graph_pass import inplaceable_npu_ops
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


def debug_time(func: Optional[Callable] = None, *, phase_name: Optional[str] = None):
    """
    A decorator / helper function that measures the execution time of another function.
    The timing is performed only when logging level is set to DEBUG.

    Args:
        func (Optional[Callable], optional): The function to be decorated/wrapped.
                                            If provided, it means this function is used as a helper function.
        phase_name (Optional[str], optional): The custom name to log for this timing phase.
                                              If not provided, the function's own name (__name__) will be used.
                                              Must be passed as a keyword argument.

    Returns:
        Callable: If used as a decorator (@debug_time(...)), returns the decorator function.
                  If used as a helper function (debug_time(func, ...)), returns the wrapped function.

    Usage:
        # As a decorator with a custom name
        @debug_time(phase_name="My Custom Phase")
        def my_function():
            pass

        # As a decorator using the function's name
        @debug_time()
        def another_function():
            pass

        # As a helper function
        def original_func():
            pass
        timed_func = debug_time(original_func, phase_name="Helper Phase Name")
    """
    def _make_wrapper(func, effective_phase_name):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger.isEnabledFor(logging.DEBUG):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                cost_time = time.perf_counter() - start_time
                logger.debug("The operation of %s phase (function name: %s) cost %.10fs.", effective_phase_name, func.__name__, cost_time)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

    def decorator(func: Callable):
        effective_phase_name = phase_name if phase_name is not None else func.__name__
        return _make_wrapper(func, effective_phase_name)

    if func is not None:
        effective_phase_name = phase_name if phase_name is not None else func.__name__
        return _make_wrapper(func, effective_phase_name)
    return decorator


def debug_compare_fx_graphs(pass_fn: Optional[Callable] = None, *, pass_name: Optional[str] = None):
    """
    A decorator / helper function that compares the FX graph after a pass optimization,
    and logging the differences including node counts, names, operations (ops), and targets.
    The comparison and timing are performed only when logging level is DEBUG.

    Args:
        pass_fn (Optional[Callable], optional): The function to be decorated/wrapped.
                                               If provided, it means this function is used as a helper function.
        pass_name (Optional[str], optional): The custom name to identify this pass in logs.
                                             If not provided, the function's own name (__name__) will be used.
                                             Must be passed as a keyword argument.

    Returns:
        Callable: If used as a decorator (@debug_compare_fx_graphs(...)), returns the decorator function.
                  If used as a helper function (debug_compare_fx_graphs(func, ...)), returns the wrapped function.

    Usage:
        # As a decorator with a custom name
        @debug_compare_fx_graphs(pass_name="My Custom Pass")
        def my_pass_function(gm: GraphModule):
            # ... perform modifications on gm ...
            return gm

        # As a decorator using the function's name
        @debug_compare_fx_graphs()
        def another_pass_function(gm: GraphModule):
            # ...
            return gm

        # As a helper function
        def original_pass_func(gm: GraphModule):
            # ...
            return gm
        wrapped_pass_func = debug_compare_fx_graphs(original_pass_func, pass_name="Helper Pass Name")
    """
    def _make_wrapper(func, effective_name):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if logger.isEnabledFor(logging.DEBUG):
                gm = next((arg for arg in args if isinstance(arg, GraphModule)), None)
                before_node_info = _get_node_info(list(gm.graph.nodes)) if gm is not None else None
                # case for self.fx_graph.graph.eliminate_dead_code
                gh = None
                if gm is None:
                    gh = getattr(func, '__self__', None)
                    before_node_info = _get_node_info(list(gh.nodes)) if gh is not None else None

                before_time = time.perf_counter()
                result = func(*args, **kwargs)
                pass_cost_time = time.perf_counter() - before_time

                after_node_info = _get_node_info(list(gm.graph.nodes)) if gm is not None else None
                if gm is None:
                    after_node_info = _get_node_info(list(gh.nodes)) if gh is not None else None
                _compare_fx_graphs(before_node_info, after_node_info, effective_name, pass_cost_time)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

    def decorator(func: Callable):
        effective_name = pass_name if pass_name is not None else func.__name__
        return _make_wrapper(func, effective_name)

    if pass_fn is not None:
        effective_name = pass_name if pass_name is not None else pass_fn.__name__
        return _make_wrapper(pass_fn, effective_name)
    return decorator


def _compare_fx_graphs(before_node_info: Dict, after_node_info: Dict, pass_name: str, pass_cost_time=0.0) -> None:
    if before_node_info is None or after_node_info is None:
        logger.warning("The before_node_info or after_node_info is None, skip compare fxgraph differences.")
        return

    headers = ["name", "cnt(before)", "cnt(after)", "diff"]

    # node name diff
    removed_names = list(set(before_node_info['node_names']) - set(after_node_info['node_names']))
    added_names = list(set(after_node_info['node_names']) - set(before_node_info['node_names']))

    # ops diff
    before_ops_dict = dict(before_node_info['node_ops'])
    after_ops_dict = dict(after_node_info['node_ops'])
    ops_diff_str = _tabulate(_merge_diff_list(before_ops_dict, after_ops_dict), headers)

    # targets diff
    before_targets_dict = before_node_info['node_targets']
    after_targets_dict = after_node_info['node_targets']
    targets_diff_str = _tabulate(_merge_diff_list(before_targets_dict, after_targets_dict), headers)

    cost_line = f"    cost {pass_cost_time:.10f}s,\n" if pass_cost_time > 0 else ""
    logger.debug(
        "After fx graph pass(%s) optimization:\n"
        "%s"
        "    before %d nodes, after %d nodes,\n"
        "    removed nodes: %s,\n"
        "    added nodes: %s,\n"
        "    operations statistics:\n %s\n\n"
        "    targets statistics:\n %s",
        pass_name,
        cost_line,
        before_node_info["total_count"],
        after_node_info["total_count"],
        removed_names,
        added_names,
        ops_diff_str,
        targets_diff_str
    )


def _get_node_info(nodes):
    total_count = len(nodes)
    node_names = []
    node_ops = []
    node_targets = []
    for node in nodes:
        node_names.append(node.name)
        node_ops.append(node.op)
        node_targets.append(node._pretty_print_target(node.target))

    return {
        'total_count': total_count,
        'node_names': node_names,
        'node_ops': Counter(node_ops),
        'node_targets': Counter(node_targets),
    }


def _tabulate(data: List[List], headers: Optional[List[str]] = None, indentation_symbol: str = " " * 4) -> str:
    num_cols = max(len(row) for row in data)
    num_cols = max(num_cols, len(headers)) if headers else num_cols
    widths = [0] * num_cols

    if headers:
        for i, header in enumerate(headers):
            if i < num_cols:
                widths[i] = max(widths[i], len(str(header)))

    for row in data:
        for i, cell in enumerate(row):
            if i < num_cols:
                cell_str = str(cell)
                widths[i] = max(widths[i], len(cell_str))

    widths = [max(w, 2) for w in widths]

    lines = []
    column_padding_symbol = " " * 2
    if headers:
        header_cells = []
        for i, (header, width) in enumerate(zip(headers, widths)):
            if i == 0:
                header_cells.append(str(header).ljust(width))
            else:
                header_cells.append(str(header).rjust(width))
        lines.append(indentation_symbol + column_padding_symbol.join(header_cells))

        separator = ["-" * w for w in widths]
        lines.append(indentation_symbol + column_padding_symbol.join(separator))

    for row in data:
        row_cells = []
        padded_row = list(row) + [""] * (num_cols - len(row))
        for i, (cell, width) in enumerate(zip(padded_row, widths)):
            cell_str = str(cell)
            if i == 0:
                row_cells.append(cell_str.ljust(width))
            else:
                row_cells.append(cell_str.rjust(width))
        lines.append(indentation_symbol + column_padding_symbol.join(row_cells))
    return "\n".join(lines)


def _merge_diff_list(before_dict, after_dict):
    merged_diff = []
    sum_before = sum(before_dict.values())
    sum_after = sum(after_dict.values())
    merged_diff.append(["total", sum_before, sum_after, sum_after - sum_before])
    all_keys = set(before_dict.keys()).union(after_dict.keys())
    for key in all_keys:
        before_value = before_dict.get(key, 0)
        after_value = after_dict.get(key, 0)
        diff_item = [key, before_value, after_value, after_value - before_value]
        merged_diff.append(diff_item)
    return merged_diff


def verify_cross_stream_event_protected(inplace_node: Node, users: List[Node]) -> bool:
    """
    Check if cross-stream users of an inplace node are protected by event synchronization.

    Supports two directions:
    - Forward: inplace happens-before user (record after inplace → wait before user)
    - Backward: user happens-before inplace (record after user → wait before inplace)

    Supports chained events via BFS propagation.

    Args:
        inplace_node: The node performing the inplace operation
        users: All downstream user nodes of the mutated input
    """
    graph = inplace_node.graph

    inplace_meta = getattr(inplace_node, 'meta', {})
    if 'stream_label' not in inplace_meta:
        return False
    inplace_stream = inplace_meta["stream_label"]

    # Identify cross-stream users
    cross_stream_users = []
    for user in users:
        user_meta = getattr(user, 'meta', {})
        user_stream = user_meta.get("stream_label")
        if user_stream != inplace_stream:
            cross_stream_users.append(user)

    if not cross_stream_users:
        logger.debug("[event_protected] No cross-stream users of %s, event protection is not required.",
                     inplace_node.name)
        return True

    # Build position map and collect event nodes in a single pass
    node_position = {}
    event_records = {}  # tag -> [(stream_label, position)]
    event_waits = {}    # tag -> [(stream_label, position)]

    event_record_targets = {
        "npugraph_ex.tagged_event_record.default",
        "npugraph_ex.tagged_event_record_on_stream.default",
    }

    event_wait_targets = {
        "npugraph_ex.tagged_event_wait.default",
        "npugraph_ex.tagged_event_wait_on_stream.default",
    }

    for pos, node in enumerate(graph.nodes):
        node_position[node] = pos

        target_str = str(node.target)
        if target_str in event_record_targets and node.args:
            tag = node.args[0]
            stream = node.meta.get("stream_label") if hasattr(node, 'meta') else None
            event_records.setdefault(tag, []).append((stream, pos))
        elif target_str in event_wait_targets and node.args:
            tag = node.args[0]
            stream = node.meta.get("stream_label") if hasattr(node, 'meta') else None
            event_waits.setdefault(tag, []).append((stream, pos))

    if not event_records or not event_waits:
        logger.debug("[event_protected] No event record/wait pairs found in graph, cannot verify protection.")
        return False

    inplace_pos = node_position.get(inplace_node, -1)

    # Forward BFS: find all (stream, position) that happen-after the inplace node.
    # happens_after[stream] = P means nodes on that stream at position > P happen-after inplace.
    happens_after = {inplace_stream: inplace_pos}
    changed = True
    while changed:
        changed = False
        for tag, records in event_records.items():
            waits = event_waits.get(tag)
            if not waits:
                continue
            for rec_stream, rec_pos in records:
                if rec_stream not in happens_after or rec_pos <= happens_after[rec_stream]:
                    continue
                for wait_stream, wait_pos in waits:
                    if wait_stream not in happens_after or wait_pos < happens_after[wait_stream]:
                        happens_after[wait_stream] = wait_pos
                        changed = True

    # Backward BFS: find all (stream, position) that happen-before the inplace node.
    # happens_before[stream] = P means nodes on that stream at position < P happen-before inplace.
    happens_before = {inplace_stream: inplace_pos}
    changed = True
    while changed:
        changed = False
        for tag, waits in event_waits.items():
            records = event_records.get(tag)
            if not records:
                continue
            for wait_stream, wait_pos in waits:
                if wait_stream not in happens_before or wait_pos >= happens_before[wait_stream]:
                    continue
                for rec_stream, rec_pos in records:
                    if rec_stream not in happens_before or rec_pos + 1 > happens_before[rec_stream]:
                        happens_before[rec_stream] = rec_pos + 1
                        changed = True

    # Verify every cross-stream user is covered by either direction
    for user in cross_stream_users:
        user_stream = getattr(user, 'meta', {}).get("stream_label")
        user_pos = node_position.get(user, -1)

        forward_ok = user_stream in happens_after and user_pos > happens_after[user_stream]
        backward_ok = user_stream in happens_before and user_pos < happens_before[user_stream]

        if not forward_ok and not backward_ok:
            logger.debug("[event_protected] Cross-stream user %s (stream=%s, pos=%d) is NOT protected "
                         "by events for inplace node %s.", user.name, user_stream, user_pos, inplace_node.name)
            return False

    logger.debug("[event_protected] All cross-stream users of %s are protected by event synchronization.",
                 inplace_node.name)
    return True
