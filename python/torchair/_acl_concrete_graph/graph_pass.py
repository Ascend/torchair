from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import Any, Optional, List, Callable, Dict, Set
import threading
import operator

import torch
from torch import fx
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

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

# try to import necessary tools from pytorch for reinplace
(
    _FunctionalizationMetadataProp,
    _get_view_inverse_node_usages,
    _get_all_later_node_usages,
    _maybe_get_inplace_op,
    _is_view_op,
    _VIEW_INVERSE_MAP,
    StorageWeakRef,
    FakeTensor,
    pytree,
    tree_map_only,
) = (None,) * 10

# flag which indicates whether torchair reinplace_with_multi_stream_check with multi-stream is available
_HAS_INTERNAL_REINPLACE_TOOL = False

try:
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.utils import _pytree as pytree
    from torch.utils._pytree import tree_map_only
    from torch.fx.passes.reinplace import (
        _FunctionalizationMetadataProp,
        _get_view_inverse_node_usages,
        _get_all_later_node_usages,
        _maybe_get_inplace_op,
        _is_view_op,
        _VIEW_INVERSE_MAP,
    )
    _HAS_INTERNAL_REINPLACE_TOOL = True
except ImportError as e:
    logger.debug(
        f"Can not import tool functions from torch.fx.passes.reinplace (Torch {torch.__version__}), "
        f"reinplace_with_multi_stream_check is not available, with error: {e}"
    )
        

aten = torch.ops.aten
# Operators that don't depend on the tensor data
META_ONLY_OPS = {
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.sym_numel.default,
    aten.sym_storage_offset.default,
}


# multi-stream checker for reinplace
def check_multi_stream_for_single_reinplace(node: torch.fx.Node) -> bool:
    """
    Check whether there are multiple streams in the current node when reinplacing.


    Note: For single reinplace operators,
          we just need to verify the 0th input support reinplaces under multiple streams
    """
    from torchair._acl_concrete_graph.utils import ReinplaceStreamChecker
    checker = ReinplaceStreamChecker()
    return checker.check_single_reinplace(node)


def check_multi_stream_for_multi_reinplace(node: torch.fx.Node) -> bool:
    """
    Check whether there are multiple streams in the current node when reinplacing.

    Note: For operators with multi_reinplace,
          we need to verify that all inputs support reinplaces under multiple streams
    """
    from torchair._acl_concrete_graph.utils import ReinplaceStreamChecker
    checker = ReinplaceStreamChecker()
    return checker.check_multi_reinplace(node)


def check_multi_stream_for_auto_functionalize(node: torch.fx.Node, mutated_arg: torch.fx.Node) -> bool:
    """
    Check whether there are multiple streams in the current node when reinplacing.

    Note: For operators with auto_functionalize scenarios.
          we need to verify that all inputs support reinplaces under multiple streams
    """
    from torchair._acl_concrete_graph.utils import ReinplaceStreamChecker
    checker = ReinplaceStreamChecker()
    return checker.check_auto_functionalize(node, mutated_arg)


@dataclass(frozen=True)
class InplaceableNpuOp:
    inplace_op: Callable[..., Any]
    mutated_arg: List[int]
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True


inplaceable_npu_ops: Dict[Callable[..., Any], InplaceableNpuOp] = {}

if hasattr(torch.ops.npu, "npu_kv_rmsnorm_rope_cache_v2"):
    inplaceable_npu_ops.update({
        torch.ops.npu.npu_kv_rmsnorm_rope_cache_v2_functional.default:
            InplaceableNpuOp(
                inplace_op=torch.ops.npu.npu_kv_rmsnorm_rope_cache_v2.default,
                mutated_arg=[5, 6],
                extra_check=check_multi_stream_for_multi_reinplace,
            )
    })

if hasattr(torch.ops.npu, "npu_mla_prolog_v3"):
    inplaceable_npu_ops.update({
        torch.ops.npu.npu_mla_prolog_v3_functional.default:
            InplaceableNpuOp(
                inplace_op=torch.ops.npu.npu_mla_prolog_v3.default,
                mutated_arg=[9, 10],
                extra_check=check_multi_stream_for_multi_reinplace,
            )
    })


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
        storage_of_reinplaced_args = set()

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
            if should_attempt_reinplace and can_inplace(node, mutated_arg) and \
                check_multi_stream_for_auto_functionalize(node, mutated_arg):
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
            if can_inplace(node, mutated_args) and inplaceable_op.extra_check(node):
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
            if can_inplace(node, mutated_arg) and check_multi_stream_for_single_reinplace(node):
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                node.target = inplace_op
            else:
                logger.debug(f"can_inplace return False, will skip reinplacing for node: {node.target}")
        elif hasattr(torch.ops.higher_order, "auto_functionalized_v2"
                     ) and node.target is torch.ops.higher_order.auto_functionalized_v2:
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
        elif hasattr(torch.ops.higher_order, "auto_functionalized"
                     ) and node.target is torch.ops.higher_order.auto_functionalized:
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


def _reinplace_inplaceable_ops_pass(gm: GraphModule, multi_stream_enabled: bool, *sample_args):
    """
    Given a fx.GraphModule, modifies it to perform "reinplacing". Just call torch.fx.passes.reinplace.
    Note: this pass can not deal with mutated inputs.
    """
    original_gm = gm

    # Set stream labels for all nodes before pattern pass
    from torchair._utils.graph_utils import add_stream_label_to_node_meta
    add_stream_label_to_node_meta(gm)

    try:
        logger.debug("[_reinplace_inplaceable_ops_pass]processing reinplace_inplaceable_ops_pass for graph: %s", id(gm))
        if _HAS_INTERNAL_REINPLACE_TOOL:
            gm = reinplace_with_multi_stream_check(gm, *sample_args)
        else:
            logger.warning_once(f"Skipped fx_pass _reinplace_inplaceable_ops_pass for unsupported fx graph {id(gm)}."
                                f"Reinplace_with_multi_stream_check is not supported.")
            return original_gm
        logger.debug("[_reinplace_inplaceable_ops_pass]End to process reinplace pass for graph: %s", id(gm))
    except NotImplementedError:
        raise
    except Exception as exception:
        if torch.__version__ < '2.5.0':
            raise RuntimeError("There is a bug in torch.fx.passes.reinplace module when torch < 2.5.0. Two possible"
                               " solutions: 1. upgrade torch version(>=2.5.0); 2. disable pass config by setting: "
                               "config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass=True") from exception
        else:
            logger.warning_once(f"Skipped fx_pass _reinplace_inplaceable_ops_pass for unsupported fx graph {id(gm)}.")
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

    # Set stream labels for all nodes before pattern pass
    from torchair._utils.graph_utils import add_stream_label_to_node_meta
    add_stream_label_to_node_meta(gm)

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


def replace_core_limit_nodes(gm: torch.fx.GraphModule):
    # use stack to handle nested scope declarations
    scope_enter_stack = []
    # record original state of current stream, revert when exit core limit scope
    core_limit_stack = []
    
    for node in gm.graph.nodes:
        if str(node.target) == "air.scope_enter.default":
            _core_limit_handle_scope_enter(node, gm, core_limit_stack, scope_enter_stack)
        elif str(node.target) == "air.scope_exit.default":
            _core_limit_handle_scope_exit(node, gm, core_limit_stack, scope_enter_stack)


def _core_limit_handle_scope_enter(node: torch.fx.Node, gm: torch.fx.GraphModule, core_limit_stack: List, scope_enter_stack: List):
    core_limit_label = ["_op_aicore_num", "_op_vectorcore_num"]
    stream_switch_label = ["_user_stream_label", "_user_stream_priority"]
    if node.args[0] == core_limit_label:
        # get current user configuration for core limit
        aicore_num, vectorcore_num = node.args[1]
        with gm.graph.inserting_before(node):
            stream_node = gm.graph.call_function(torch.npu.current_stream)
            core_get_node = gm.graph.call_function(torch.npu.get_stream_limit, args=(stream_node,))
            aicore_num_node = gm.graph.call_function(operator.getitem, args=(core_get_node, 'cube_core_num'))
            vectorcore_num_node = gm.graph.call_function(operator.getitem, args=(core_get_node, 'vector_core_num'))
            core_set_node = gm.graph.call_function(torch.npu.set_stream_limit, args=((stream_node, int(aicore_num), 
                                                                                    int(vectorcore_num))))
        # core limit scope enter node can be replaced by set stream limit node
        node.replace_all_uses_with(core_set_node)
        gm.graph.erase_node(node)
        # record current stream, original core states for rolling back
        core_limit_stack.append([stream_node, aicore_num_node, vectorcore_num_node, aicore_num, vectorcore_num])
        scope_enter_stack.append("core_limit")
    elif node.args[0] == stream_switch_label:
        # do nothing if current stream switch is not within a core limit scope
        if not core_limit_stack:
            scope_enter_stack.append("other")
            return
        # use core configuration from last core limit scope
        _, _, _, last_aicore_num, last_vectorcore_num = core_limit_stack[-1]
        # stream switch scope enter node need to be retained
        with gm.graph.inserting_after(node):
            stream_node = gm.graph.call_function(torch.npu.current_stream)
        with gm.graph.inserting_after(stream_node):
            core_get_node = gm.graph.call_function(torch.npu.get_stream_limit, args=(stream_node,))
        with gm.graph.inserting_after(core_get_node):
            gm.graph.call_function(torch.npu.set_stream_limit, args=((stream_node, int(last_aicore_num), int(last_vectorcore_num))))
            vectorcore_num_node = gm.graph.call_function(operator.getitem, args=(core_get_node, 'vector_core_num'))
            aicore_num_node = gm.graph.call_function(operator.getitem, args=(core_get_node, 'cube_core_num'))
        core_limit_stack.append([stream_node, aicore_num_node, vectorcore_num_node, last_aicore_num, last_vectorcore_num])
        scope_enter_stack.append("stream_switch")
    else:
        # still push current scope enter into stack to match forward scope exits
        scope_enter_stack.append("other")


def _core_limit_handle_scope_exit(node: torch.fx.Node, gm: torch.fx.GraphModule, core_limit_stack: List, scope_enter_stack: List):
    last_scope_enter_op = scope_enter_stack.pop()
    if last_scope_enter_op == "other":
        return
    # retrieve current stream, original core states when exit scope
    cur_stream, original_aicore_num, original_vectorcore_num, _, _ = core_limit_stack.pop()
    with gm.graph.inserting_after(node):
        core_set_node = gm.graph.call_function(torch.npu.set_stream_limit, args=(cur_stream, original_aicore_num, original_vectorcore_num))
    if last_scope_enter_op == "core_limit":
        node.replace_all_uses_with(core_set_node)
        gm.graph.erase_node(node)


def reinplace_with_multi_stream_check(gm, *sample_args):
    """
    Stream-aware reinplace pass.

    This is identical to the original reinplace implementation, except that
    inplace transformations are conservatively disallowed when aliasing uses
    occur across different streams.
    
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
    We look for out-of-place op call sites like `b = a.add(...)`,
    and convert them to be inplace (`b = a.add_(...)`),
    as long as the input to the current operator ("a") isn't re-used
    anywhere later in the graph.

    This pass currently expects to operate on a **functional, ATen** graph.
    This can be obtained by running `make_fx(functionalize(f))`.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can't reinplace node `b = a.add(...)` if "a" aliases any of the
    inputs to the program.

    Given a node "b = foo(a, args...) the algorithm for re-inplacing is as follows:

    (1) Perform some initial checks on the metadata of "a" and "args..."
        that can disqualify them from being reinplaced.

      (1a) Check that the self argument we're attempting to reinplace
           has acceptable dtype/size metadata to reinplace with.

           For example, if we have:
             a = torch.ones(1)
             b = torch.ones(10)
             out = torch.add(a, b)
           We can't turn that into
             a.add_(b)
           Because that would require resizing "a".

           Similarly, we can't convert torch.ge(a, b) into a.ge_(b),
           because that would require changing a's dtype (from e.g. float32 to bool).
           Note that in this specific example, we could technically do better..

           If we see the pattern:
             a_1 = a.ge(b)
             a_2 = aten._to_copy(a_1, a.dtype)
           Then we this should be valid to completely re-inplace
           (this is exactly what functionalization will emit when it sees a.ge_(b)).

           This optimization is only really important for user programs
           that directly use inplace comparison ops though.

           We also cannot re-inplace on tensors that have overlapping memory,
           e.g. torch.ones(1).expand(4, 4).add_(1)

      (1b) Check if "a" is an alias of any of the program inputs.

          If it is, skip and move to the next node.
          Inplace'ing an op that would cause it to mutate a program is not sound,
          because that would be a side effect visible to the user.

          NOTE: there's a future optimization that we should make:
          if "a" is a (alias of a)  program input, but later in the program
          there is a node that looks like "a.copy_(...)",
          Then re-inplacing is ok to do - we are temporarily re-using a's buffer,
          which will later be overwritten by the copy_() call.

          This will be an important optimization to have for programs that mutate
          their inputs. It currently isn't implemented though.

      (1c) Check if "a" and "args..." alias

          For example, re-inplacing to create code like the below
          isn't guaranteed to be sound:

            aten.mul_(a, a)

    (2) Check that "a" and all of its outstanding aliases are not used anywhere
        later in the graph. If this is the case, then it's safe to re-inplace
        to "b = foo_(a)".

        There are a few caveats to this, explained in more detail below:
        (a) If "a" is used later as an argument to a view op, that is okay.
            It's only a problem if "a" (or that view) is later passed
            into a normal operator, or if it is returned as the program output.
        (b) If "a" is a repeat argument in `foo()`, then don't reinplace.
            Most ATen kernels don't make any guarantees that this is sound,
            e.g. if you do aten.mul_(a, a).
            So we'll just ban re-inplacing in this case.
            It's only a problem if "a" (or that view) is later passed
        (c) If "a" is used as an input into a view "inverse" / "scatter"
            operator, it is potentially fine to re-inplace
            (and remove that scatter operator from the graph).
            See below for a more detailed example.

        NOTE: there is an optimization in this step that is crucial
        to fully recovering performance from functionalization.

        Given this program:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a)
            torch.ops.aten.fill_(b, 0)
            return d

        Functionalization will emit the following:
        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a, 0, 1)
            b_updated = torch.ops.aten.fill(b, 0)
            a_updated = torch.ops.aten.diagonal_scatter(a, b_updated, 0, 1)
            return a_updated

        Ordinarily, we would not be able to reinplace the fill,
        because "b" aliases with "a" which is used by the diagonal_scatter call.

        "re-inplacing" is on the hook for figuring out that it is ok to
        completely, the expensive diagonal_scatter call, if we re-inplace the add().

        So, for every `alias in alias_set(a)`, instead of checking
        that "alias" is not used anywhere later in the graph,
        we check that
            EITHER:
          (a) alias is not used anywhere later in the graph
            OR:
          (b) alias is used exactly once later on in the graph,
              in the following op:

                out = foo_scatter(alias, x, args...)

              where the following must hold:
                (i) "foo_scatter" is the "inverse" operator for foo.
                    This only applies to "foo" ops that are view operators,
                    which view into a subset of the original tensor's memory.
                    In practice, there are ~4 operators where this applies:
                      diagonal -> diagonal_scatter
                      slice -> slice_scatter
                      select -> select_scatter
                      as_strided -> as_strided_scatter
                (ii) "args..." are the same between the foo() and foo_scatter() calls.

    (3) Perform the actual re-inplacing on foo!

      (3b) is the common case, but special care is needed for {view}_scatter (3a)

      (3a) {view}_scatter ops.

        Consider this program:
          a = torch.zeros(2, 2)
          b = torch.ones(2)
          a[0] = b

        Post functionalization, that will look like:
          a = torch.zeros(2)
          b = torch.ones(1)
          a_updated = torch.select_scatter(a, b, 0, 0)

        In this case though, there is no "functional" op to re-inplace!
        Instead, we'd like to directly remove toe select_scatter call.
        We already know from (3) that this is valid,
        because "a" has no later usages in the graph.

        We perform the re-inplacing on the {view}_scatter op like so
        Before:
          a_updated = torch.select_scatter(a, b, args...)
        After:
          a_slice = a.select(a, args...)
          a_slice.copy_(b)

      (3b) Otherwise, replace the functional op with its inplace variant.
        Before:
          b = foo(a, args...)
        After:
          a.foo_(args...)

    (4) Finally, after converting either:
          Before:
            b = foo(a)
          After:
            foo_(a)
        or
          Before:
            b = {slice}_scatter(a, mutated_slice, args...)
          After:
            slice = {slice}(a, args...)
            slice.copy_(mutated_slice)

        We now need to find all later nodes that use "b" as an argument
        and update them to take in "a" instead.

        Note that for the majority of inplace ops, this isn't actually necessary
        (because most inplace ops return "self" as their output).
        This isn't generally true for all mutable ops though, which is why
        we need to actually replace all of the arguments.

        We also need to update our metadata of Dict[StorageWeakRef, Set[Node]],
        That maps a given tensor storage to the set of all nodes that take in that storage
        as an input.
        Specifically, re-inplacing `b = foo(a)` causes "a" and "b"'s sets to get fused
        together.

    (5) Any "view_inverse/scatter" nodes that were identified as "it's ok to ignore them"
        during step (3) get manually deleted from the graph.
        Their outputs are no longer used, so technically standard DCE would be able
        to do this, but we can no longer run FX's DCE pass now that we have mutable
        ops in the graph.
    """
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)

    """
    Useful debug printing
    def _print(x):
    if isinstance(x, FakeTensor):
    print(f'fake_result: {StorageWeakRef(x._typed_storage()).cdata}')

    for n in gm.graph.nodes:
    print(n.format_node())
    if hasattr(n, 'meta'):
    print(f'node_idx: {n.meta["node_idx"]}')
    if 'fake_result' in n.meta:
    tree_map(_print, n.meta['fake_result'])
    if 'view_of' in n.meta:
    print(f'view_of: {str(n.meta["view_of"])}')
    print()
    """

    # We need to know which nodes correspond to inputs (or their aliases)
    # so we know not to re-inplace them.
    # NOTE: later, we'll need to add an optimization for fully recovering performance
    # on programs that mutate inputs.
    input_storages = {
        StorageWeakRef(node.meta["fake_result"]._typed_storage())
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.meta["fake_result"], torch.Tensor)
        )
    }

    # We also need to know for a given node, what are all of its aliasing nodes.
    storage_to_nodes: dict[StorageWeakRef, set[torch.fx.Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if "fake_result" in n.meta:
            # Tree-mapping because some ops can return lists of tensors.
            def _add_to_map(x):
                if isinstance(x, FakeTensor):
                    storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)

            pytree.tree_map_(_add_to_map, n.meta["fake_result"])
    
    # inplace-ify functional ops, subject to the constraints written below.
    all_later_view_inverse_nodes_to_delete = set()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            # Today, the re-inplace pass on directly acts on:
            # - functional ops with an inplace variant
            # - {view}_scatter ops that can be potentially removed from the graph.
            # Both of these ops take in tensor first args, so filtering on this condition
            # makes the later code simpler.
            # We should revisit this at some point though, particularly when we also want
            # the reinplacer to be able to handle out= and mutable operators
            # and tensorlist first args (like `_foreach_` ops).
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if len(node.target._schema.arguments) < 1:
                continue
            if type(node.target._schema.arguments[0].type) != torch.TensorType:
                continue
            # ---- Step 1a: metadata checks ----
            # Step 1a: Check that the self argument we're attempting to reinplace
            # has the same size/stride as the output.
            # For example, we shouldn't try to reinplace torch.add(scalar_tensor, larger_tensor)
            # As it would require resizing scalar_tensor.
            # (We could potentially swizzle this into larger_tensor.add_(scalar_tensor),
            # this is probably an optimization to revisit later).
            self_arg = node.args[0]
            self_flattened = pytree.tree_leaves(self_arg.meta["fake_result"])
            node_flattened = pytree.tree_leaves(node.meta["fake_result"])    
            self_has_wrong_metadata = False
            if len(self_flattened) == len(node_flattened):
                for self_meta, node_meta in zip(self_flattened, node_flattened):
                    if self_meta.numel() != node_meta.numel():
                        self_has_wrong_metadata = True
                    if self_meta.dtype != node_meta.dtype:
                        self_has_wrong_metadata = True
                    # We also cannot re-inplace on tensors that have internal memory overlap.
                    # e.g. torch.ones(1).expand(4, 4).add_(1)
                    if torch._debug_has_internal_overlap(self_meta) == 1:
                        self_has_wrong_metadata = True
            # Here, we (optimistically) assume that a.resize(b) is valid to re-inplace,
            # Since users should never really be calling the functional "torch.ops.aten.resize"
            # op directly in their programs.
            if self_has_wrong_metadata and node.target != torch.ops.aten.resize.default:
                continue
            # ---- Step 1b: do not mutate program inputs ----
            # Step 1b: ensure that the op we're trying to re-inplace isn't a program input
            self_arg_storage = StorageWeakRef(
                self_arg.meta["fake_result"]._typed_storage()
            )
            if self_arg_storage in input_storages:
                # TODO: later, add the optimization for handling `copy_()` calls in the graph.
                continue
            # ---- Step 1c: repeated self argument ----
            if len([x for x in node.args if x is self_arg]) > 1:
                # Step 1c:
                # Calling stuff like aten.mul_(a, a) isn't guaranteed to be sound,
                # so we prevent re-inplacing in this case.
                continue
            self_arg_storage = StorageWeakRef(
                self_arg.meta["fake_result"]._typed_storage()
            )
            self_aliases = storage_to_nodes[self_arg_storage]
            # ---- Step 2: find later aliasing usages (stream-aware) ----
            # First, we find all later usages of any of the aliases of self_arg.
            later_node_usages = _get_all_later_node_usages(
                self_aliases, node.meta["node_idx"]
            )
    
            later_view_inverse_node_usages = _get_view_inverse_node_usages(
                later_node_usages, self_aliases
            )
    
            can_reinplace = (
                len(later_node_usages - later_view_inverse_node_usages) == 0
            )
            # [TORCHAIR] Do multi-stream check.
            if not can_reinplace:
                check_stream = False
            else:
                check_stream = check_multi_stream_for_single_reinplace(node)
            logger.debug("Node[%s] check reinplace is %s, check multi-stream is %s",
                         node.name, can_reinplace, check_stream)
            if not can_reinplace or not check_stream:
                continue
    
            # ---- Step 3a: handle view_scatter ----
            # Step 3a: Special handling for when we see *_scatter operators.
            # When we see an operator like `b = torch.slice_scatter(a, ...)`,
            # instead of trying to "inplace" it into a.slice_scatter_(..._),
            # we would prefer to remove it from the graph entirely,
            # and instead copy_() the slice directly into the larger tensor.
            # See the description of the algorithm for a full example.
            if (
                node.target in _VIEW_INVERSE_MAP
                and node not in all_later_view_inverse_nodes_to_delete
            ):
                view_op = _VIEW_INVERSE_MAP[node.target]
                # Before:
                #   base_updated = torch.ops.aten.slice_scatter.default(base, mutated_slice, args...)
                # After:
                #   slice = torch.ops.aten.slice.default(base, args...)
                #   slice.copy_(mutated_slice)
                with gm.graph.inserting_before(node):
                    mutated_slice_node = node.args[1]
                    remaining_slice_args = node.args[2:]
                    slice_node = gm.graph.create_node(
                        "call_function",
                        view_op,
                        (self_arg,) + tuple(remaining_slice_args),
                        node.kwargs,
                    )
                    gm.graph.create_node(
                        "call_function",
                        torch.ops.aten.copy_.default,
                        (
                            slice_node,
                            mutated_slice_node,
                        ),
                        {},
                    )
                # Add the slice_scatter node to our "nodes to delete" list.
                all_later_view_inverse_nodes_to_delete.add(node)
            
            else:
                # Step 3b: Check to see if this operator has an inplace variant.
                maybe_inplace_op = _maybe_get_inplace_op(node.target)
                if maybe_inplace_op is None:
                    continue
                # And if so, replace it with its inplace variant.
                node.target = maybe_inplace_op
            # ---- Step 4: update alias maps ----
            # At this point, 'storage_to_nodes' will be stale.
            # Now that we're inplacing `b = foo(a)`, we need to effectively
            # union together the dict values for b and a's storage.
            # Hmm... morally I think we also want to keep the `fake_result` metadata
            # up to date here, but I'm not sure how easy it is to do.
            # Maybe it's fine to wait until the end of the pass to update it.
            curr_node_storage = StorageWeakRef(
                node.meta["fake_result"]._typed_storage()
            )
            storage_to_nodes[self_arg_storage].update(
                storage_to_nodes[curr_node_storage]
            )
            storage_to_nodes[curr_node_storage].update(
                storage_to_nodes[self_arg_storage]
            )
            
            # Need to remember the view_scatter view nodes we found so we can remove them alter.
            all_later_view_inverse_nodes_to_delete.update(
                later_view_inverse_node_usages
            )
    
            # Step 4:
            # Now that we've replaced b = a.foo() with a.foo_(),
            # We need to replace any later usages of "b" with "a"
            for old in itertools.chain([node], later_view_inverse_node_usages):
                new = old.args[0]
                nodes_to_update = [
                    n for n in old.users if n.meta["node_idx"] > node.meta["node_idx"]
                ]
                for node_to_update in nodes_to_update:
    
                    def replace_arg(a):
                        if a == old:
                            return new
                        return a
                    
                    # First, replace usages of "b" with "a"
                    node_to_update.args = tree_map_only(
                        torch.fx.Node, replace_arg, node_to_update.args
                    )
                    node_to_update.kwargs = tree_map_only(
                        torch.fx.Node, replace_arg, node_to_update.kwargs
                    )
                    
                    # Second, update our storage_to_nodes data structure.
                    old_flattened_res = pytree.tree_leaves(old.meta["fake_result"])
                    node_flattened_res = pytree.tree_leaves(
                        node_to_update.meta["fake_result"]
                    )
    
                    old_res_storage = {
                        StorageWeakRef(x._typed_storage())
                        for x in old_flattened_res
                        if isinstance(x, FakeTensor)
                    }
                    node_res_storage = {
                        StorageWeakRef(x._typed_storage())
                        for x in node_flattened_res
                        if isinstance(x, FakeTensor)
                    }
                    
                    # This will happen if we're updating a view op, e.g.
                    # e.g. replacing
                    #     x = view(old)
                    #     x = view(new)
                    # When that happens, we need to make sure to keep our
                    # storage mapping up to date.
                    #
                    # We're checking for len(...) == 1 here because all view ops are guaranteed to return either a single tensor,
                    # or multiple tensors that all share the same storage.
                    # We can't just check equality because we might encounter FX nodes that return zero tensor outputs.
                    if (
                        len(old_res_storage) == 1
                        and len(node_res_storage) == 1
                        and old_res_storage == node_res_storage
                    ):
                        new_flattened_res = pytree.tree_leaves(new.meta["fake_result"])
                        new_res_storage = {
                            StorageWeakRef(x._typed_storage())
                            for x in new_flattened_res
                            if isinstance(x, FakeTensor)
                        }
                        (new_ref,) = new_res_storage
                        (node_ref,) = node_res_storage
                        # Technically, "old_ref" and all its aliases will remain
                        # in our mapping.
                        # That should be fine though, since we deleted "old"
                        # from the graph at this point.
                        storage_to_nodes[node_ref].update(
                            storage_to_nodes[new_ref]
                        )
                        storage_to_nodes[new_ref].update(
                            storage_to_nodes[node_ref]
                        )

    # ---- Step 5: delete eliminated view_scatter nodes ----
    # Need to take care not to delete any of these nodes until after *all* modifications
    # to the graph are finished.
    for to_delete in all_later_view_inverse_nodes_to_delete:
        gm.graph.erase_node(to_delete)

    gm.recompile()
    return gm
