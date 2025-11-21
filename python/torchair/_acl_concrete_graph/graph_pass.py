from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, List, Callable, Dict, Set
import threading
import operator

import torch
from torch import fx
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.fx.passes.reinplace import reinplace, _maybe_get_inplace_op, _is_view_op

from torchair.core.utils import logger

try:
    from torch._C._dynamo.guards import compute_overlapping_tensors
except ImportError:
    compute_overlapping_tensors = None
    logger.debug("function[compute_overlapping_tensors] is not support on torch < 2.6")

try:
    from torch._dynamo.utils import ReinplaceCounters, ReInplaceTrigger
except ImportError:
    ReinplaceCounters = None
    ReInplaceTrigger = None
    logger.debug("function[ReinplaceCounters] and function[ReInplaceTrigger] is not support on torch < 2.6")

try:
    from torch.utils._ordered_set import OrderedSet
except Exception:
    logger.debug("function[OrderedSet] is not support on torch < 2.6")


aten = torch.ops.aten


@dataclass(frozen=True)
class InplaceableNpuOp:
    inplace_op: Callable[..., Any]
    mutated_arg: List[int]
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True


# Operators that don't depend on the tensor data
META_ONLY_OPS = {
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.sym_numel.default,
    aten.sym_storage_offset.default,
}

inplaceable_npu_ops: Dict[Callable[..., Any], InplaceableNpuOp] = {}
try:
    npu = torch.ops.npu
    inplaceable_npu_functional_ops = {
        npu.npu_kv_rmsnorm_rope_cache_v2_functional.default: InplaceableNpuOp(
            npu.npu_kv_rmsnorm_rope_cache_v2.default, [5, 6]
        ),
    }
    inplaceable_npu_ops.update(inplaceable_npu_functional_ops)
except AttributeError as e:
    logger.debug(f"cannot update npu ops with AttributeError: {e}")


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
            if src.target == operator.getitem and (
                src.args[0].target in inplaceable_npu_ops
            ):
                src = src.args[0]
            copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst] = node
            to_replace_targets.add(node.args[1])
    logger.debug(f"[_reinplace_input_mutated_ops] mutated input replace candidates: {to_replace_targets=}, "
                 f"{copy_args_to_copy_nodes=}")

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
    
    def log_inplace_results(
        node_name,
        old_tensors_to_clone,
        tensors_to_clone,
        missed_args,
        missed_nodes,
        trigger,
    ):
        # Total size of possibly_missed_reinplacing_opportunities for tensors with static shapes.
        missed_bytes = 0

        def node_bytes(node):
            t = node.meta.get("val", None)
            if (
                t is not None
                and isinstance(t.element_size(), int)
                and isinstance(t.numel(), int)
            ):
                return t.element_size() * t.numel()
            else:
                return 0

        for node in missed_nodes:
            if isinstance(node, (list, tuple)):
                for n in node:
                    missed_bytes += node_bytes(n)
            else:
                missed_bytes += node_bytes(node)

        logger.info(
            "For node %s, attempted to reinplace %s. We were unable to reinplace %s; "
            "%s (if non-empty) are possible missed reinplacing opportunities that may be bad for "
            "memory usage and performance. Total size of missed opportunities with static shapes is"
            " : %s bytes.",
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_bytes,
        )

        ReinplaceCounters.add_missed_opportunities(trigger, len(missed_args))
        ReinplaceCounters.add_missed_bytes(trigger, missed_bytes)

    replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}

    def reinplace_and_refine_tensors_to_clone(
        old_tensors_to_clone, kwargs, node_name, trigger
    ):
        tensors_to_clone: list[str] = []
        storage_of_reinplaced_args = OrderedSet[int | None]()

        # Those used to count possibly_missed_reinplacing_opportunities
        missed_nodes = []
        missed_args = []

        def tensor_with_same_storage_already_reinplaced(arg):
            if isinstance(arg, (list, tuple)):
                return any(
                    get_node_storage(a) in storage_of_reinplaced_args for a in arg
                )
            return get_node_storage(mutated_arg) in storage_of_reinplaced_args

        for arg in old_tensors_to_clone:
            if arg not in kwargs:
                raise KeyError(f"Missing required key {arg} in kwargs")

            mutated_arg = kwargs[arg]

            # Let's say we have:
            # - op(x, y) that mutates both x and y
            # - new_x, new_y = functional_op(x, y) is the functional variant
            # If we are presented with functional_op(x, x), we must not reinplace
            # this into op(x, x), because then it would be writing to the same Tensor.
            # Instead, it's OK to reinplace one of them and to clone the other:
            # >>> y = x.clone()
            # >>> op(x, y)
            # This also applies if we have views: functional_op(x, x[0])
            # should not reinplace into op(x, x[0]).
            should_attempt_reinplace = not tensor_with_same_storage_already_reinplaced(
                mutated_arg
            )
            if should_attempt_reinplace and can_inplace(node, mutated_arg):
                # In general, we probably do not need those optimizations.
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                if trigger != ReInplaceTrigger.AUTO_FUNC_V2:
                    for user in node.users:
                        # For auto_functionalize_v2, arg is the index of the base, where base at index i corresponds to
                        # output atindex size(out)+i.
                        # This used to compare string with integers before for auto_functionalize_v2. Not sure
                        # if it was needed for inplaceable_triton_ops?
                        if user.target is operator.getitem and user.args[1] == arg:
                            replace_dict[user] = mutated_arg

                if isinstance(mutated_arg, (list, tuple)):
                    for a in mutated_arg:
                        storage_of_reinplaced_args.add(get_node_storage(a))
                else:
                    storage_of_reinplaced_args.add(get_node_storage(mutated_arg))
            else:
                if should_attempt_reinplace:
                    missed_args.append(arg)
                    missed_nodes.append(mutated_arg)

                tensors_to_clone.append(arg)

        log_inplace_results(
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_nodes,
            trigger,
        )
        return tensors_to_clone

    for node in graph.nodes:
        inplaceable_op = inplaceable_npu_ops.get(node.target, None)
        if inplaceable_op is not None:
            # Here we properly remove copy epilogues for
            # ops that mutate multiple inputs.
            mutated_args = [node.args[arg_index] for arg_index in inplaceable_op.mutated_arg]
            if not all((arg, node) in copy_args_to_copy_nodes for arg in mutated_args):
                logger.debug(f"reinplace failed, all mutated args(get_item) must have copy epilogues: {node.target}")
                continue
            if can_inplace(node, mutated_args):
                for arg in mutated_args:
                    copy_node = copy_args_to_copy_nodes[(arg, node)]
                    replace_dict[copy_node] = copy_node.args[0]
                    if copy_node.args[1].target == operator.getitem:
                        replace_dict[copy_node.args[1]] = copy_node.args[0]
                node.target = inplaceable_op.inplace_op
            else:
                logger.debug(f"can_inplace return False, will skip reinplacing for node: {node.target}")
        elif node in to_replace_targets:
            mutated_arg = node.args[0]
            inplace_op = _maybe_get_inplace_op(node.target)
            if inplace_op is None:
                logger.debug("cannot find an inplace op for node %s", node.target)
                continue
            if can_inplace(node, mutated_arg):
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                node.target = inplace_op
            else:
                logger.debug(f"can_inplace return False, will skip reinplacing for node: {node.target}")
        elif hasattr(torch.ops.higher_order, "auto_functionalized_v2") and node.target is torch.ops.higher_order.auto_functionalized_v2:
            _mutable_op = node.args[0]
            kwargs = node.kwargs

            all_bases = kwargs["_all_bases"]
            bases_to_clone = range(len(all_bases))
            base_tensors_dct = dict(enumerate(all_bases))
            new_bases_to_clone: list[int] = reinplace_and_refine_tensors_to_clone(
                bases_to_clone,
                base_tensors_dct,
                node.target,
                ReInplaceTrigger.AUTO_FUNC_V2,
            )
            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = new_bases_to_clone
        elif hasattr(torch.ops.higher_order, "auto_functionalized") and node.target is torch.ops.higher_order.auto_functionalized:
            _mutable_op = node.args[0]
            from torch._higher_order_ops.auto_functionalize import get_mutable_args

            tensors_to_clone, _ = get_mutable_args(_mutable_op)
            # Don't try to reinplace Tensor | None args that are None.
            tensors_to_clone = [
                t for t in tensors_to_clone if node.kwargs[t] is not None
            ]
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                tensors_to_clone,
                node.kwargs,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V1,
            )

            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = tensors_to_clone
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
    original_gm = gm
    try:
        logger.debug("[_reinplace_inplaceable_ops_pass]processing reinplace_inplaceable_ops_pass for graph: %s", id(gm))
        gm = reinplace(gm, *sample_args)
        logger.debug("End to process reinplace inplaceable ops fx pass for graph: %s", id(gm))
    except NotImplementedError:
        raise
    except Exception as exception:
        if torch.__version__ < '2.5.0':
            raise RuntimeError("There is a bug in torch.fx.passes.reinplace module when torch < 2.5.0. Two possible"
                               " solutions: 1. upgrade torch version(>=2.5.0); 2. disable pass config by setting: "
                               "config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass=True") from exception
        else:
            logger.warning_once(f"Skipped fx_pass torch.fx.passes.reinplace for unsupported fx graph {id(gm)}.")
            return original_gm
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


_GLOBAL_SCOPE_TAG_TO_EVENT = {}
_GLOBAL_EVENT_LOCK = threading.Lock()


def _create_event_by_name(name: str):
    global _GLOBAL_SCOPE_TAG_TO_EVENT
    with _GLOBAL_EVENT_LOCK:
        if name not in _GLOBAL_SCOPE_TAG_TO_EVENT.keys():
            new_event = torch.npu.Event()
            _GLOBAL_SCOPE_TAG_TO_EVENT[name] = new_event
            logger.debug(f"[Multi-stream] Created new event {new_event} with key '{name}'.")


def apply_event_closure_with_multi_stream(graph_module: fx.GraphModule, graph_name: str, tagged_event_names: List[str],
                                          user_stream_label: Set[str]):
    stream_scope_enter_nodes = []
    stream_scope_exit_nodes = []
    stream_scope_enter_nodes_dict = {}
    stream_scope_exit_nodes_list = []
    scope_enter_nodes_stack: List[fx.Node] = []
    for node in graph_module.graph.nodes:
        if str(node.target) == "air.scope_enter.default":
            node.kwargs = {**node.kwargs, 'need_execute': True}
            if len(node.args) > 0 and '_user_stream_label' in node.args[0]:
                stream_scope_enter_nodes.append(node)
                # When 'args[0]' of air.scope_enter.default includes the string '_user_stream_label',
                # there must be an 'args[1]' to store value associated with that key.
                # We store that value for later stream switch.
                # Use a set for storage to eliminate duplicate values.
                user_stream_label.add(node.args[1][0])
                stream_scope_enter_nodes_dict[node.name] = node.args[1][0]
            scope_enter_nodes_stack.append(node)
        elif str(node.target) == "air.scope_exit.default":
            node.kwargs = {**node.kwargs, 'need_execute': True}
            if (len(scope_enter_nodes_stack) > 0 and len(scope_enter_nodes_stack[-1].args) > 0 and
                    '_user_stream_label' in scope_enter_nodes_stack[-1].args[0]):
                stream_scope_exit_nodes.append(node)
                stream_scope_exit_nodes_list.append(node.name)
            scope_enter_nodes_stack.pop()
    if len(stream_scope_enter_nodes) == 0:
        logger.debug("No scope_enter node found in graph[%s], no need to insert event.", id(graph_module))
        return False, stream_scope_enter_nodes_dict, stream_scope_exit_nodes_list

    # These imports are needed for torch.ops.air.tagged_event_record/wait.default to work.
    import torchair
    from torchair.ops._tagged_event import _npu_create_tagged_event
    first_node = next(iter(graph_module.graph.nodes))
    enter_tag = graph_name + '_' + first_node.name
    _create_event_by_name(enter_tag)
    tagged_event_names.append(enter_tag)
    # Insert event record before graph input, insert event wait after scope_enter node
    with graph_module.graph.inserting_before(first_node):
        graph_module.graph.call_function(torch.ops.air.tagged_event_record.default, args=(enter_tag, True))
    for node in stream_scope_enter_nodes:
        with graph_module.graph.inserting_after(node):
            graph_module.graph.call_function(torch.ops.air.tagged_event_wait.default, args=(enter_tag, True))

    output_node = list(graph_module.graph.nodes)[-1]
    if output_node is None or output_node.op != "output":
        raise RuntimeError(f"Graph must have output node as last node, but got {output_node}")
    for node in stream_scope_exit_nodes:
        # insert event record after scope_exit node, insert event wait before graph output
        exit_tag = graph_name + '_' + node.name
        _create_event_by_name(exit_tag)
        tagged_event_names.append(exit_tag)
        with graph_module.graph.inserting_before(node):
            graph_module.graph.call_function(torch.ops.air.tagged_event_record.default, args=(exit_tag, True))
        with graph_module.graph.inserting_before(output_node):
            graph_module.graph.call_function(torch.ops.air.tagged_event_wait.default, args=(exit_tag, True))

    graph_module.graph.lint()
    logger.debug("End to insert event in graph[%s].", id(graph_module))
    logger.debug("Tagged event names are: [%s]", tagged_event_names)
    logger.debug("user_stream_label names are: [%s]", user_stream_label)
    return len(stream_scope_enter_nodes) > 0, stream_scope_enter_nodes_dict, stream_scope_exit_nodes_list


def apply_event_record(graph_module: fx.GraphModule):
    wait_record_dic = {}
    for node in graph_module.graph.nodes:
        if str(node.target) == "air.wait.default":
            new_args = _insert_record_nodes(graph_module, node, wait_record_dic)
            node.args = (new_args,)
    graph_module.graph.lint()
    logger.debug("End insert record node in graph:[%s]", id(graph_module))


def _insert_record_nodes(graph_module, node, wait_record_dic):
    new_args = []
    for wait_node in node.args[0]:
        if str(wait_node.target) == "air.record.default":
            new_args.append(wait_node)
        elif wait_node in wait_record_dic:
            new_args.append(wait_record_dic[wait_node])
        else:
            with graph_module.graph.inserting_after(wait_node):
                node = graph_module.graph.call_function(torch.ops.air.record.default, args=())
                new_args.append(node)
                wait_record_dic[wait_node] = node
    return new_args
