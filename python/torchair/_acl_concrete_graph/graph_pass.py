from collections import defaultdict
from typing import Any, Optional

import torch
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.fx.passes.reinplace import reinplace, _maybe_get_inplace_op, _is_view_op

from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger

try:
    from torch._C._dynamo.guards import compute_overlapping_tensors
except ImportError:
    compute_overlapping_tensors = None
    logger.debug("function[compute_overlapping_tensors] is not support on torch < 2.6")

aten = torch.ops.aten

# Operators that don't depend on the tensor data
META_ONLY_OPS = {
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.sym_numel.default,
    aten.sym_storage_offset.default,
}


def get_storage(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


def get_node_storage(node: torch.fx.Node) -> Optional[int]:
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])


def _mutated_input_reinplace(gm: GraphModule) -> GraphModule:
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.
    """

    graph = gm.graph
    copy_args_to_copy_nodes = {}
    # maps argument to the first copy_ node that mutates it.
    copy_nodes = {}
    to_replace_targets = set()
    storage_to_nodes = defaultdict(list)
    node_order: dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target == aten.copy_.default and node.args[0].op in (
                "placeholder",
                "get_attr",
        ):
            dst = node.args[0]
            src = node.args[1]
            copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst] = node
            to_replace_targets.add(node.args[1])

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node, mutated_arg):
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        def is_meta_only_user(node):
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            return node.target in META_ONLY_OPS

        for view in shared_view_nodes:
            for user in view.users:
                user_loc = node_order[user]
                # Skip all users before node
                if user_loc <= node_loc:
                    continue
                # Ignore uses after the copy_ epilogue node, where the input
                # has already been mutated anyway
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # Reinplacing does not change shape metadata
                if is_meta_only_user(user):
                    continue
                # If our graph looks like:
                # foo(mutated_arg)
                # mutated_arg.copy_(other)
                # then it's safe for us to reinplace foo because mutated_arg
                # will get overwritten anyways.
                if (
                        user.target is torch.ops.aten.copy_.default
                        and mutated_arg is user.args[0]
                ):
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        # ls should be a list of tensors that all shares the same storage.
        def _overlap(ls) -> bool:
            try:
                if compute_overlapping_tensors is not None:
                    return len(compute_overlapping_tensors(ls)) != 0
                else:
                    return True
            except GuardOnDataDependentSymNode:
                # If we fail with data dependent error we assume they all overlap.
                return True

        if isinstance(mutated_arg, (list, tuple)):
            # TODO Using _overlap here causes a several issues.
            unique_storages = set(get_node_storage(arg) for arg in mutated_arg)
            if len(unique_storages) != len(mutated_arg):
                # At least two Tensors in mutated_arg alias each other, so we can't reinplace it.
                # We can probably do better (that is, reinplace one of them and clone the other)
                # but that requires more work and mutable List[Tensor] are not that common.
                return False
            return all(can_inplace(node, arg) for arg in mutated_arg)

        if get_node_storage(mutated_arg) is None:
            return False

        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]

        # Only keep tensor that might overlap with mutated_arg.
        shared_view_nodes = [
            v
            for v in shared_view_nodes
            if _overlap([mutated_arg.meta["val"], v.meta["val"]])
        ]

        if mutated_arg.op in ("placeholder", "get_attr"):
            # Get the first copy_ node that mutates the mutated_arg.
            copy_node = copy_nodes.get(mutated_arg, None)
            if copy_node is None:
                # There is no copy_ back to the candidate mutated_arg (which is a graph input).
                # Therefore the semantics of the program are that it does not mutate
                # mutated_arg, so we cannot re-inplace it.
                return False
            if any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=copy_node, mutated_arg=mutated_arg
            ):
                return False

            return True
        elif any(view.op in ("placeholder", "get_attr") for view in shared_view_nodes):
            # This should never happen in auto_functionalize_v2 non-inference mode,
            # since all mutated_arg are bases.

            # If mutated arg is view of any of the inputs of the graph,
            # do not allow for inplacing.
            # This would require more sophisticated algorithm to handle
            return False
        else:
            return not any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=None, mutated_arg=mutated_arg
            )

    replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in graph.nodes:
        if node in to_replace_targets:
            mutated_arg = node.args[0]
            inplace_op = _maybe_get_inplace_op(node.target)
            if inplace_op is None:
                logger.debug("cannot find an inplace op for node %s", node.target)
                continue
            if can_inplace(node, mutated_arg):
                # TODO: this doesn't properly remove copy epilogues for
                # ops that mutate multiple inputs. Need to revise the copy
                # node tracking logic to support the case.
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                node.target = inplace_op
    for node, replacement in replace_dict.items():
        while replacement in replace_dict:
            replacement = replace_dict[replacement]
        replace_dict[node] = replacement

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


def _reinplace_inplaceable_ops_pass(gm: GraphModule, *sample_args):
    """
    Given a fx.GraphModule, modifies it to perform "reinplacing". Just call torch.fx.passes.reinplace.
    Note: this pass can not deal with mutated inputs.
    """
    try:
        logger.debug("[_reinplace_inplaceable_ops_pass]processing reinplace_inplaceable_ops_pass for graph: %s", id(gm))
        gm = reinplace(gm, *sample_args)
        logger.debug("End to process reinplace inplaceable ops fx pass for graph: %s", id(gm))
    except NotImplementedError as e:
        raise e
    except Exception as e:
        raise RuntimeError("There is a bug in torch.fx.passes.reinplace module when torch < 2.5.0. Two possible"
                           " solutions: 1. upgrade torch version(>=2.5.0); 2. disable config by setting: "
                           "config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass=True") from e
    return gm


def _reinplace_input_mutated_ops(gm: GraphModule):
    """
    Given a fx.GraphModule, modifies it to perform "reinplacing", mutating the nodes of the graph.
        We try to handle the mutated input reinplace by reusing inductor fx pass for reinplacing input mutated ops,
        which is able to replace different op names(usually, in-place op and out-place op differ only
        slightly in their names. Specifically, the in-place op names are appended with a "_" compared to their
        out-place version) and not-first mutated args.
    """
    logger.debug("[_reinplace_input_mutated_ops]processing reinplace_input_mutated_ops_pass for graph: %s", id(gm))
    _mutated_input_reinplace(gm)
    logger.debug("End to process reinplace input mutated ops fx pass for graph: %s", id(gm))
    return gm
