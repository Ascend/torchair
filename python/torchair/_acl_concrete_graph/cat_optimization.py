from typing import List, Optional, Tuple, Any, Union

import operator as op_mod

import torch
from torch.fx import GraphModule
from torchair.core.utils import logger
from torchair._utils.graph_utils import debug_compare_fx_graphs
from torchair._acl_concrete_graph.acl_graph import (
    is_constant,
    is_sym,
    have_sym_in_meta,
    construct_fx_node_shape,
)


def _build_sym_inputs_from_placeholders(gm: GraphModule) -> dict:
    """
    Build symbol -> placeholder mapping. When all SymInts come from model inputs
    (e.g. arg0_1: Sym(s77), arg1_1: Sym(s27)), we can use placeholders to replace them.
    """
    sym_inputs = {}
    for node in gm.graph.nodes:
        if node.op == 'placeholder' and 'val' in node.meta:
            val = node.meta['val']
            if is_sym(val):
                sym_inputs[val.node.expr] = node
    return sym_inputs


def _get_out_variant(op_target) -> Optional[Any]:
    if hasattr(op_target, '_overloadpacket'):
        return getattr(op_target._overloadpacket, 'out', None)
    return None


def _has_out_variant(op_target) -> bool:
    return _get_out_variant(op_target) is not None


def _validate_cat_node(cat_node: torch.fx.Node,
                       cat_node_set: set) -> Tuple[bool, Optional[str]]:
    if len(cat_node.args) < 1:
        return False, "Missing arguments"

    tensors = cat_node.args[0]
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 2:
        return False, "Need at least 2 tensors"

    dim = cat_node.args[1] if len(cat_node.args) >= 2 else cat_node.kwargs.get('dim', 0)
    if not isinstance(dim, int) or dim != 0:
        return False, f"Only supports dim=0, got dim={dim}"

    if 'val' not in cat_node.meta:
        return False, "Missing output metadata"

    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.fx.Node) or tensor.op != 'call_function':
            return False, f"Input {i} is not a function call node"
        if 'val' not in tensor.meta:
            return False, f"Input {i} missing metadata"
        if not _has_out_variant(tensor.target):
            return False, f"Input {i} op {tensor.target} has no .out variant"
        for user in tensor.users:
            if user is not cat_node and user in cat_node_set:
                return False, f"Input {i} is shared with another cat node"
        if i > 0:
            prev_shape = tensors[i - 1].meta['val'].shape
            curr_shape = tensor.meta['val'].shape
            if len(prev_shape) != len(curr_shape):
                return False, "Incompatible tensor ranks"
            if any(prev_shape[d] != curr_shape[d] for d in range(1, len(curr_shape))):
                return False, "Shape mismatch on non-concat dimensions"

    return True, None


def _create_slice_node(gm: GraphModule, output_tensor: torch.fx.Node,
                       offset: Union[int, torch.fx.Node],
                       size: Union[int, torch.fx.Node]) -> torch.fx.Node:
    """Create slice node at dim=0. Caller must set the insertion context."""
    if is_constant(offset) and is_constant(size):
        end = int(offset) + int(size)
        return gm.graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(output_tensor, 0, offset, end)
        )
    add_node = gm.graph.call_function(op_mod.add, args=(offset, size))
    return gm.graph.call_function(
        torch.ops.aten.slice.Tensor,
        args=(output_tensor, 0, offset, add_node)
    )


def _create_out_op_node(gm: GraphModule, input_node: torch.fx.Node,
                        slice_node: torch.fx.Node) -> torch.fx.Node:
    """Create op node with .out parameter, preserving original kwargs and meta."""
    out_op = _get_out_variant(input_node.target)
    kwargs = dict(input_node.kwargs) if input_node.kwargs else {}
    kwargs['out'] = slice_node
    target = out_op if out_op is not None else input_node.target
    new_node = gm.graph.call_function(target, args=input_node.args, kwargs=kwargs)
    new_node.meta.update(input_node.meta)
    return new_node


@debug_compare_fx_graphs(pass_name="remove_cat_ops")
def optimize_cat_with_out_tensor(gm: GraphModule, config=None) -> GraphModule:
    cat_nodes = [n for n in gm.graph.nodes
                 if n.op == 'call_function' and n.target == torch.ops.aten.cat.default]

    if not cat_nodes:
        return gm

    from torchair._utils.graph_utils import add_stream_label_to_node_meta
    add_stream_label_to_node_meta(gm)

    logger.debug(f"[remove_cat_ops] Found {len(cat_nodes)} cat node(s)")

    cat_node_set = set(cat_nodes)
    optimized_count = 0

    for cat_node in cat_nodes:
        can_opt, reason = _validate_cat_node(cat_node, cat_node_set)
        if not can_opt:
            logger.debug(f"Skip {cat_node.name}: {reason}")
            continue

        tensors = list(cat_node.args[0])
        output_meta = cat_node.meta['val']
        logger.debug(f"Optimizing {cat_node.name} ({len(tensors)} inputs)")

        sym_inputs = _build_sym_inputs_from_placeholders(gm) if have_sym_in_meta(output_meta) else {}
        node_order = {n: idx for idx, n in enumerate(gm.graph.nodes)}
        first_input = min(tensors, key=lambda t: node_order[t])
        if have_sym_in_meta(output_meta):
            with gm.graph.inserting_before(first_input):
                output_shape = tuple(construct_fx_node_shape(
                    list(output_meta.shape), sym_inputs, id(gm)
                ))
        else:
            output_shape = output_meta.shape

        sizes_resolved = []
        for input_node in tensors:
            input_meta = input_node.meta.get('val')
            if input_meta is None:
                break
            size = input_meta.shape[0]
            if is_constant(size):
                sizes_resolved.append(int(size))
            elif sym_inputs:
                with gm.graph.inserting_before(first_input):
                    size_node = construct_fx_node_shape([size], sym_inputs, id(gm))[0]
                sizes_resolved.append(size_node)
            else:
                break
        if len(sizes_resolved) != len(tensors):
            continue

        cat_stream = cat_node.meta.get('stream_label')
        with gm.graph.inserting_before(first_input):
            if cat_stream is not None:
                gm.graph.call_function(
                    torch.ops.air.scope_enter.default,
                    args=(['_user_stream_label'], [cat_stream])
                )
            output_tensor = gm.graph.call_function(
                torch.ops.aten.empty.memory_format,
                args=(output_shape,),
                kwargs={
                    'dtype': output_meta.dtype,
                    'device': output_meta.device,
                    'pin_memory': False,
                    'memory_format': None
                }
            )
            if cat_stream is not None:
                gm.graph.call_function(torch.ops.air.scope_exit.default, args=())

        output_tensor.meta['val'] = output_meta
        output_tensor.meta['stream_label'] = cat_stream

        offsets = [0]
        last_offset_node = output_tensor
        for i in range(1, len(tensors)):
            prev_offset = offsets[-1]
            size = sizes_resolved[i - 1]
            if is_constant(prev_offset) and is_constant(size):
                offsets.append(int(prev_offset) + int(size))
            else:
                with gm.graph.inserting_after(last_offset_node):
                    offset_node = gm.graph.call_function(op_mod.add, args=(prev_offset, size))
                offsets.append(offset_node)
                last_offset_node = offset_node

        replacement_map = {}
        for i, input_node in enumerate(tensors):
            size = sizes_resolved[i]
            offset = offsets[i]
            with gm.graph.inserting_before(input_node):
                slice_node = _create_slice_node(gm, output_tensor, offset, size)
                out_op_node = _create_out_op_node(gm, input_node, slice_node)
            replacement_map[input_node] = out_op_node

        for old_node, new_node in replacement_map.items():
            old_node.replace_all_uses_with(new_node)
        cat_node.replace_all_uses_with(output_tensor)

        for node in reversed(list(replacement_map.keys()) + [cat_node]):
            if node in gm.graph.nodes and len(node.users) == 0:
                node.meta.clear()
                gm.graph.erase_node(node)

        cat_node_set.discard(cat_node)
        optimized_count += 1

    if optimized_count > 0:
        gm.graph.lint()
        gm.recompile()
        logger.debug(f"remove_cat_ops: Optimized {optimized_count} cat node(s)")
    return gm